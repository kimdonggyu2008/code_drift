"""Exercise the real CLI, checkpoint reload, and scope contract on CPU."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch
import yaml

from dac.model import Discriminator
from tests.test_codedrift import tiny_dac


def test_cli_train_resume_and_reject_changed_scope(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    torch.manual_seed(17)
    generator_path, discriminator_path = tmp_path / "g.pth", tmp_path / "d.pth"
    tiny_dac().save(generator_path, package=False)
    disc = Discriminator(periods=[2], rates=[], fft_sizes=[])
    disc.save(discriminator_path, package=False)
    for split in ("train", "val"):
        folder = tmp_path / split
        folder.mkdir()
        wave = np.random.default_rng(3).normal(0, .05, 4410).astype(np.float32)
        sf.write(folder / "input.wav", wave, 44100, subtype="FLOAT")
    config = {
        "$include": ["conf/code_drift/code.yml"],
        "train/build_dataset.folders": {"speech": [str(tmp_path / "train")]},
        "val/build_dataset.folders": {"speech": [str(tmp_path / "val")]},
        "train/AudioDataset.duration": .05, "val/AudioDataset.duration": .05,
        "train/AudioDataset.n_examples": 4, "val/AudioDataset.n_examples": 1,
        "batch_size": 1, "val_batch_size": 1, "num_workers": 0,
        "sample_freq": 1, "valid_freq": 1, "val_idx": [0],
        "code_drift.scope": "encoder",
        "MultiScaleSTFTLoss.window_lengths": [64, 128],
        "MelSpectrogramLoss.window_lengths": [64, 128],
        "MelSpectrogramLoss.n_mels": [8, 16],
        "MelSpectrogramLoss.mel_fmin": [0, 0],
        "MelSpectrogramLoss.mel_fmax": [None, None],
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config))
    save_path = tmp_path / "run"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONPATH=str(repo), WANDB_MODE="disabled")
    base_cmd = [sys.executable, "-m", "scripts.train", "--args.load", str(config_path),
                "--save_path", str(save_path)]

    def run(extra, success=True):
        result = subprocess.run(base_cmd + extra, cwd=repo, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
        assert (result.returncode == 0) == success, result.stdout[-8000:]
        return result.stdout

    run(["--pretrained_generator", str(generator_path), "--pretrained_discriminator",
         str(discriminator_path), "--num_iters", "1"])
    latest = save_path / "latest" / "dac"
    before = torch.load(latest / "weights.pth", weights_only=False)["state_dict"]
    assert torch.load(latest / "tracker.pth", weights_only=False)["step"] == 1
    meta = torch.load(latest / "metadata.pth", weights_only=False)
    assert meta["completed_updates"] == 1
    assert meta["code_drift"]["scope"] == "encoder"

    run(["--resume", "--num_iters", "2"])
    assert torch.load(latest / "tracker.pth", weights_only=False)["step"] == 2
    after = torch.load(latest / "weights.pth", weights_only=False)["state_dict"]
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("encoder."))
    assert all(torch.equal(before[k], after[k]) for k in before if not k.startswith("encoder."))
    assert "already complete" in run(["--resume", "--num_iters", "2"])
    assert "Resume Code Drift settings differ" in run(
        ["--resume", "--num_iters", "3", "--code_drift.scope", "decoder"], success=False
    )
    # A deliberately new weights-only run resets optimizer/tracker and may change scope.
    run(["--resume", "--load_weights", "--num_iters", "1", "--code_drift.scope", "decoder"])
    assert torch.load(latest / "tracker.pth", weights_only=False)["step"] == 1
    restarted = torch.load(latest / "weights.pth", weights_only=False)["state_dict"]
    assert all(torch.equal(after[k], restarted[k]) for k in after if not k.startswith("decoder."))
    assert any(not torch.equal(after[k], restarted[k]) for k in after if k.startswith("decoder."))
