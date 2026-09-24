# Our approach, with a little maths

We want to find out whether a small change helps Bodhan transcribe our Marathi recordings more accurately.

The **encoder** turns speech into information the model can use. The **decoder** uses that information and the preceding text to predict the next piece of text.

```text
Recording → encoder → decoder → transcript
```

During training, we compare predictions with the correct transcript. The resulting error guides the weight updates. All our approaches keep the full encoder frozen: it still processes audio, but its weights do not change.

**A: change selected decoder weights.** Start from the original Bodhan model. Update the last two decoder layers and the final decoder normalization, which adjusts the scale of their output. Keep the other weights frozen. This is our existing approach. Those layers are a starting choice, not a proven best choice for Marathi.

**B: add LoRA to the decoder.** Start independently from the original model. Freeze all original weights. Add small trainable matrices to every decoder attention block, covering attention to earlier text and to audio.

For one matrix, the idea is:

$$
z = Wx + \frac{\alpha}{r}VUx
$$

`W` is the unchanged original matrix. `U` and `V` learn an added adjustment. `r` controls their size; `alpha` controls its scale. Our starting settings are rank 8 and alpha 16. This usually means fewer trainable numbers, but the original model still needs memory.

A and B differ in both **how** weights change and **where** changes happen. Their comparison cannot isolate LoRA alone.

**C: add LoRA after A, only if A beats B.** If A has strictly lower normalized validation WER than B, load A's best saved model, freeze its weights, and train fresh LoRA matrices in the same places as B. Skip C on a tie. C gets extra training, so any improvement might partly come from that extra work.

**Choose first, then test.** Word error rate is:

$$
WER = \frac{\text{wrong words + missing words + extra words}}{\text{words in the correct transcript}}
$$

Lower is better. Validation recordings choose checkpoints, trigger C, and select among the original model, A, B, and C. Save that decision before evaluating all candidates on separate test recordings. The original model can win. We have no comparison results yet.

See [README.md](README.md) for commands and settings.
