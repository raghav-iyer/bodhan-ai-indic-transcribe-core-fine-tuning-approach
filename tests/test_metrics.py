import pytest

from marathi_asr.metrics import aggregate, edit_distance, paired_comparison, score_pair
from marathi_asr.text import clean_transcript, normalize_for_scoring


def row(key, reference, hypothesis):
    return {"id": key, "reference": reference, "prediction": hypothesis, "audio_sha256": key,
            "scores": score_pair(reference, hypothesis)}


def test_marathi_normalization_keeps_marks_and_numbers():
    assert normalize_for_scoring("  महाराष्ट्रात,  पाऊस आहे। २०२६! ") == "महाराष्ट्रात पाऊस आहे २०२६"
    assert clean_transcript(" नमस्कार\nमहाराष्ट्र। ") == "नमस्कार महाराष्ट्र।"
    assert normalize_for_scoring("Hello, WORLD!") == "hello world"


def test_corpus_wer_not_mean_of_utterance_wer():
    rows = [row("a", "एक", ""), row("b", "दोन तीन चार पाच", "दोन तीन चार पाच")]
    assert aggregate(rows)["normalized"]["wer"] == 0.2
    assert aggregate([row("a", "एक", "दोन तीन चार")])["normalized"]["wer"] == 3.0


def test_empty_hypothesis_and_insertions():
    assert edit_distance(["a", "b"], []) == 2
    assert edit_distance([], ["a", "b"]) == 2
    assert aggregate([row("a", "नमस्कार", "")])["normalized"]["cer"] == 1
    with pytest.raises(ValueError):
        aggregate([])
    with pytest.raises(ValueError):
        aggregate([row("a", "!!!", "नमस्कार")])


def test_raw_versus_normalized():
    result = score_pair("नमस्कार।", "नमस्कार")
    assert result["raw"]["word_errors"] == 1
    assert result["normalized"]["word_errors"] == 0


def test_paired_bootstrap_deterministic_and_direction():
    a = [row("a", "एक दोन", "एक तीन"), row("b", "नमस्कार", "")]
    b = [row("a", "एक दोन", "एक दोन"), row("b", "नमस्कार", "नमस्कार")]
    result = paired_comparison(a, b, samples=50)
    assert result == paired_comparison(a, b, samples=50)
    assert result["normalized_wer_delta_tuned_minus_baseline"] == -2 / 3
    assert result["paired_bootstrap_95pct_ci"][1] < 0


@pytest.mark.parametrize("change", ["id", "reference", "audio_sha256"])
def test_mismatched_comparison_rejected(change):
    a = row("a", "एक दोन", "एक दोन")
    b = {**a, change: "different"}
    with pytest.raises(ValueError):
        paired_comparison([a], [b])


def test_duplicate_predictions_rejected():
    a = row("a", "एक", "एक")
    with pytest.raises(ValueError):
        paired_comparison([a, a], [a, a])
