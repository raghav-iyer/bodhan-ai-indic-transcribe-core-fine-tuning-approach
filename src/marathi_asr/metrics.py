"""Count transcript errors and compare models on the same recordings."""

import random

from .text import clean_transcript, normalize_for_scoring


def edit_distance(reference, hypothesis):
    """Minimum insertions, deletions and substitutions needed to match two sequences."""
    # Each row stores the best answer for prefixes of the two sequences.
    # Keeping only two rows saves memory compared with storing the whole table.
    previous_row = list(range(len(hypothesis) + 1))
    for reference_index, reference_item in enumerate(reference, 1):
        current_row = [reference_index]
        for hypothesis_index, hypothesis_item in enumerate(hypothesis, 1):
            insertion = current_row[-1] + 1
            deletion = previous_row[hypothesis_index] + 1
            substitution = previous_row[hypothesis_index - 1] + (reference_item != hypothesis_item)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    return previous_row[-1]


def score_pair(reference, hypothesis):
    """Count word and character errors, with and without scoring cleanup."""
    scores = {}
    text_policies = (("raw", clean_transcript), ("normalized", normalize_for_scoring))
    for style, transform in text_policies:
        reference_text = transform(reference)
        hypothesis_text = transform(hypothesis)
        reference_words = reference_text.split()
        hypothesis_words = hypothesis_text.split()
        # CER counts Unicode code points; spaces are excluded.
        reference_characters = list(reference_text.replace(" ", ""))
        hypothesis_characters = list(hypothesis_text.replace(" ", ""))
        scores[style] = {
            "word_errors": edit_distance(reference_words, hypothesis_words),
            "reference_words": len(reference_words),
            "char_errors": edit_distance(reference_characters, hypothesis_characters),
            "reference_chars": len(reference_characters),
        }
    return scores


def aggregate(rows):
    """Add errors across recordings before dividing; do not average individual WERs."""
    if not rows:
        raise ValueError("Cannot score an empty corpus")
    result = {"utterances": len(rows), "cer_unit": "Unicode code points excluding spaces"}
    count_names = ("word_errors", "reference_words", "char_errors", "reference_chars")
    for style in ("raw", "normalized"):
        totals = {}
        for name in count_names:
            totals[name] = sum(row["scores"][style][name] for row in rows)
        if not totals["reference_words"] or not totals["reference_chars"]:
            raise ValueError("References are empty after normalization")
        result[style] = {
            **totals,
            "wer": totals["word_errors"] / totals["reference_words"],
            "cer": totals["char_errors"] / totals["reference_chars"],
        }
    return result


def paired_comparison(baseline, tuned, samples=1000, seed=42):
    """Estimate the WER change and uncertainty using matched recordings.

    For each bootstrap sample, draw recording pairs with replacement. Both models
    always receive the same sampled recordings, so the comparison stays paired.
    """
    baseline_by_id = {row["id"]: row for row in baseline}
    tuned_by_id = {row["id"]: row for row in tuned}
    if len(baseline_by_id) != len(baseline) or len(tuned_by_id) != len(tuned):
        raise ValueError("Duplicate prediction IDs")
    if not baseline_by_id or baseline_by_id.keys() != tuned_by_id.keys():
        raise ValueError("Baseline and tuned predictions must cover the identical nonempty test set")

    score_pairs = []
    for recording_id in sorted(baseline_by_id):
        baseline_row = baseline_by_id[recording_id]
        tuned_row = tuned_by_id[recording_id]
        same_text = baseline_row["reference"] == tuned_row["reference"]
        same_audio = baseline_row["audio_sha256"] == tuned_row["audio_sha256"]
        if not same_text or not same_audio:
            raise ValueError(f"Mismatched evaluation example {recording_id}")
        # Recalculate from text so stale saved scores cannot affect the result.
        baseline_score = score_pair(baseline_row["reference"], baseline_row["prediction"])["normalized"]
        tuned_score = score_pair(tuned_row["reference"], tuned_row["prediction"])["normalized"]
        score_pairs.append((baseline_score, tuned_score))

    reference_words = sum(base["reference_words"] for base, _ in score_pairs)
    error_change = sum(tuned["word_errors"] - base["word_errors"] for base, tuned in score_pairs)
    wer_change = error_change / reference_words

    random_generator = random.Random(seed)
    sampled_changes = []
    for _ in range(samples):
        sampled_pairs = [random_generator.choice(score_pairs) for _ in score_pairs]
        sampled_word_count = sum(base["reference_words"] for base, _ in sampled_pairs)
        if sampled_word_count:
            sampled_error_change = sum(
                tuned["word_errors"] - base["word_errors"] for base, tuned in sampled_pairs
            )
            sampled_changes.append(sampled_error_change / sampled_word_count)

    sampled_changes.sort()
    interval = None
    if sampled_changes:
        last_index = len(sampled_changes) - 1
        interval = [sampled_changes[int(0.025 * last_index)], sampled_changes[int(0.975 * last_index)]]
    return {
        "normalized_wer_delta_tuned_minus_baseline": wer_change,
        "paired_bootstrap_95pct_ci": interval,
        "bootstrap_samples": samples,
        "seed": seed,
        "interpretation": "Negative delta favors fine-tuned model",
        "caveat": "Utterance bootstrap ignores dependence between recordings of the same sentence/speaker.",
    }
