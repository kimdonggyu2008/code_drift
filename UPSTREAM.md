# Upstream provenance

- Repository: https://github.com/descriptinc/descript-audio-codec
- Branch inspected: `main`
- Pinned commit: [`c7cfc5d2647e26471dc394f95846a0830e7bec34`](https://github.com/descriptinc/descript-audio-codec/commit/c7cfc5d2647e26471dc394f95846a0830e7bec34)
- Imported on: 2026-10-31
- Original MIT license: retained verbatim in `LICENSE`.
- Original README: retained verbatim in `README_DAC.md`.

This repository is a GitHub fork of `descriptinc/descript-audio-codec`.
The upstream history, initial source import, and Code Drift commits are retained.
The earlier independent repository is preserved as
[`code_drift-import-backup-20261031`](https://github.com/kimdonggyu2008/code_drift-import-backup-20261031).
The main README contains the original DAC README with a short Code Drift notice.

## Changes relative to the pinned source

- `dac/nn/codedrift.py`: objectives, reduction, frozen-RVQ and E/D scope configuration.
- `dac/model/dac.py`: opt-in training-only forward branch; no state-dict key changes.
- `dac/model/discriminator.py`: retain constructor settings as attributes so
  weights-only checkpoints reload the actual discriminator configuration.
- `scripts/train.py`: pretrained weight inputs, configuration, additive loss,
  logging, trainable parameter selection, resume-setting checks. Checkpoints count
  completed updates; resume starts at the next update.
- `conf/code_drift/`: examples based on the official 44.1 kHz configuration.
- `tests/test_codedrift*.py`: CPU gradient/update and CLI-resume regression checks.
- Documentation and local-artifact ignore rules.

Encoder, decoder, discriminator architecture, RVQ search/residual/straight-through estimator,
base loss implementations, and upstream utilities are retained. There is no new
trainable module or inference-time loss.

The paper is [arXiv:2410.11025v2 §IV](https://arxiv.org/html/2410.11025v2#S4).
Its recipe and this open-checkpoint example are not identical; explicitly chosen
details are recorded in `docs/implementation.md`.
