# Architecture and Compatibility Notes

Technical reference for Kokoro-82M fine-tuning compatibility.

For how-to training steps, use `TRAINING_GUIDE.md`.

## Kokoro-82M Component Layout

Reference component sizes used for checkpoint compatibility checks:

| Component | Parameters |
|---|---|
| bert (PLBERT) | 6.29M |
| bert_encoder | 0.39M |
| predictor | 16.19M |
| text_encoder | 5.61M |
| decoder (ISTFTNet) | 53.28M |
| Total | 81.76M |

Voicepack target shape:
- `[510, 1, 256]` (float32)

## G2P Notes

- G2P backend: `misaki` + `espeak-ng`

## Sequence Length Constraint

- PLBERT max position embeddings: 512
- Practical training cap: 510 cleaned tokens

Samples above this should be filtered before batching.

## Inference Packaging Notes

When exporting trained checkpoints for `KModel`, ensure the expected components are present and keys align with Kokoro inference code:
- `bert`
- `bert_encoder`
- `predictor`
- `text_encoder`
- `decoder`

Use `scripts/test_inference.py` to verify conversion and produce sample outputs.
