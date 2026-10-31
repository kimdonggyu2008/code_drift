# Validation — 2026-10-31

Checked on CPU without using the GPU occupied by the ongoing research run.

## Executed checks

- 30 tests in `tests/test_codedrift.py`: unsquared channel L2, hard-target detach,
  three objectives versus an independent official-DAC trace, nine loss/scope
  gradient combinations, actual parameter updates and frozen RVQ state,
  disabled-mode forward/RNG parity, zero-weight bypass, invalid settings,
  resume contract, four real training-loop updates with all five original
  weighted losses, and five example configurations.
- `tests/test_codedrift_cli.py`: real entry point with temporary synthetic audio,
  save at update 1, restart in a new process to update 2, no repeated update
  after completion, frozen-module persistence, rejection of a changed scope, and
  a deliberate weights-only restart with reset optimizer/tracker.
- Official public DAC44 checkpoint on a synthetic 220 Hz, 16,758-sample input:
  exact agreement with the existing research helper for all three loss values,
  the reconstructed waveform and first-pass codes. See `loss_parity.json`.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest tests/test_codedrift.py tests/test_codedrift_cli.py -q
```

These validate implementation and wiring, not audio-quality improvement, full 50k
training reproduction, GPU/AMP/DDP behavior, or bitwise continuation of the data
stream. Upstream `tests/test_train.py` launches GPU training and was not run.

The ongoing research run retains its frozen source and checkpoints and has not
been switched to this new trainer.
