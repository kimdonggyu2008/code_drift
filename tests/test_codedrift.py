"""CPU tests: actual DAC RVQ/STE, gradient routing, and official train_loop."""

from types import SimpleNamespace

import pytest
import torch
from audiotools import AudioSignal, ml

from dac import DAC
from dac.model import Discriminator
from dac.nn import loss as losses
from dac.nn.codedrift import CodeDriftConfig, channel_l2_mean, representation_loss


@pytest.fixture(autouse=True)
def reproducible_cpu():
    torch.manual_seed(9)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_dac():
    return DAC(encoder_dim=8, encoder_rates=[2, 2], latent_dim=32,
               decoder_dim=32, decoder_rates=[2, 2], n_codebooks=2,
               codebook_size=16, codebook_dim=4, quantizer_dropout=0.5)


def snapshot(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def changed(before, model, prefix):
    return any(not torch.equal(before[k], v) for k, v in model.state_dict().items()
               if k.startswith(prefix + "."))


def test_reduction_and_stop_gradient():
    left = torch.tensor([[[3., 0.], [4., 0.]]], requires_grad=True)
    right = torch.zeros_like(left, requires_grad=True)
    assert channel_l2_mean(left, right).item() == 2.5  # NOT MSE or squared L2
    enc = representation_loss("enc", right, None, None, left, None)
    assert torch.autograd.grad(enc, right, retain_graph=True)[0].abs().sum() > 0
    code = representation_loss("code", None, None, right, None, left)
    assert torch.autograd.grad(code, right, allow_unused=True, retain_graph=True)[0] is None
    assert torch.autograd.grad(code, left)[0].abs().sum() > 0
    with pytest.raises(ValueError, match="identical"):
        channel_l2_mean(left, right[..., :1])


@pytest.mark.parametrize("kind", ["enc", "proj", "code"])
def test_forward_matches_independent_official_trace(kind):
    model = tiny_dac()
    cfg = CodeDriftConfig(True, kind, 1., "both")
    cfg.configure(model)
    x = torch.randn(2, 1, 1023)  # exercises padding and reconstructed-wave crop
    z = model.encoder(model.preprocess(x, model.sample_rate))
    zq, ids, projected, commitment, book = model.quantizer(z)
    y = model.decode(zq)[..., :x.shape[-1]]
    z2 = model.encoder(model.preprocess(y, model.sample_rate))
    _, _, projected2, _, _ = model.quantizer(z2)
    selected = torch.cat([q.codebook(ids[:, i]).transpose(1, 2)
                          for i, q in enumerate(model.quantizer.quantizers)], 1)
    left, right = {"enc": (z2, z), "proj": (projected2, projected),
                   "code": (projected2, selected)}[kind]
    expected = ((left - right).square().sum(1).sqrt()).mean()
    out = model(x, model.sample_rate, **cfg.forward_kwargs())
    assert torch.equal(out["audio"], y)
    assert torch.equal(out["codes"], ids)
    assert torch.equal(out["latents"], projected)
    assert torch.equal(out["vq/commitment_loss"], commitment)
    assert torch.equal(out["vq/codebook_loss"], book)
    torch.testing.assert_close(out["codedrift/loss"], expected)


@pytest.mark.parametrize("kind", ["enc", "proj", "code"])
@pytest.mark.parametrize("scope", ["encoder", "decoder", "both"])
def test_auxiliary_gradients_and_actual_frozen_update(kind, scope):
    model = tiny_dac()
    cfg = CodeDriftConfig(True, kind, 1., scope)
    cfg.configure(model)
    model.train()
    cfg.train_mode(model)
    before = snapshot(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    out = model(torch.randn(1, 1, 1023), **cfg.forward_kwargs())
    out["codedrift/loss"].backward()
    for name in ("encoder", "decoder", "quantizer"):
        grads = [p.grad for p in getattr(model, name).parameters()]
        active = name == scope or (scope == "both" and name != "quantizer")
        if active:
            assert sum(g.abs().sum() for g in grads if g is not None) > 0
        else:
            assert all(g is None for g in grads)
    optimizer.step()
    assert not changed(before, model, "quantizer")
    assert changed(before, model, "encoder") == (scope in ("encoder", "both"))
    assert changed(before, model, "decoder") == (scope in ("decoder", "both"))


def test_disabled_is_original_dac_forward_and_zero_weight_skips_return():
    model = tiny_dac().train()
    before = snapshot(model)
    cfg = CodeDriftConfig()
    cfg.configure(model)
    assert all(p.requires_grad for p in model.parameters())
    x = torch.randn(2, 1, 1023)
    rng = torch.get_rng_state()
    z, codes, projected, commitment, book = model.encode(model.preprocess(x, 44100))
    original = {"audio": model.decode(z)[..., :1023], "z": z, "codes": codes,
                "latents": projected, "vq/commitment_loss": commitment, "vq/codebook_loss": book}
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    out = model(x, **cfg.forward_kwargs())
    assert out.keys() == original.keys()
    assert all(torch.equal(out[k], v) for k, v in original.items())
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())

    cfg = CodeDriftConfig(True, "code", 0., "both")
    cfg.configure(model)
    calls = []
    handle = model.encoder.register_forward_hook(lambda *a: calls.append(1))
    out = model(x, **cfg.forward_kwargs())
    handle.remove()
    assert len(calls) == 1 and "codedrift/loss" not in out


@pytest.mark.parametrize("kwargs", [
    {"loss": "unknown"}, {"weight": float("nan")}, {"weight": -1},
    {"enabled": True, "weight": 1}, {"scope": "rvq"}, {"loss": "code"},
])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        CodeDriftConfig(**kwargs)


def test_resume_contract_rejects_changed_scope_or_weight():
    cfg = CodeDriftConfig(True, "code", 1., "encoder")
    cfg.check_resume({"code_drift": cfg.to_dict()})
    for other in (CodeDriftConfig(True, "code", 1., "decoder"),
                  CodeDriftConfig(True, "code", 10., "encoder"), CodeDriftConfig()):
        with pytest.raises(ValueError, match="Resume"):
            other.check_resume({"code_drift": cfg.to_dict()})
    with pytest.raises(ValueError):
        cfg.check_resume({})


@pytest.mark.parametrize("scope,kind", [
    ("encoder", "code"), ("decoder", "code"), ("both", "code"), ("both", "none")
])
def test_official_train_loop_keeps_base_losses_and_updates_correct_modules(scope, kind):
    from scripts import train
    model = tiny_dac()
    cfg = CodeDriftConfig(True, kind, 1. if kind != "none" else 0., scope)
    cfg.configure(model)
    disc = Discriminator(rates=[], periods=[2], fft_sizes=[])
    opt_g = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4)
    state = train.State(
        generator=model, optimizer_g=opt_g, scheduler_g=torch.optim.lr_scheduler.ExponentialLR(opt_g, 1.),
        discriminator=disc, optimizer_d=opt_d, scheduler_d=torch.optim.lr_scheduler.ExponentialLR(opt_d, 1.),
        stft_loss=losses.MultiScaleSTFTLoss(window_lengths=[64, 128]),
        mel_loss=losses.MelSpectrogramLoss(n_mels=[8, 16], window_lengths=[64, 128]),
        gan_loss=losses.GANLoss(disc), waveform_loss=losses.L1Loss(),
        train_data=SimpleNamespace(transform=lambda signal, **kw: signal),
        val_data=None, tracker=None, code_drift=cfg,
    )
    weights = {"mel/loss": 15., "adv/gen_loss": 1., "adv/feat_loss": 2.,
               "vq/commitment_loss": .25, "vq/codebook_loss": 1.}
    before = snapshot(model)
    before_disc = snapshot(disc)
    # CUDA_VISIBLE_DEVICES='' is used by the documented test command.
    accel = ml.Accelerator()
    assert accel.device == "cpu", "Run these tests with CUDA_VISIBLE_DEVICES=''"
    batch = {"signal": AudioSignal(torch.randn(1, 1, 2047), 44100),
             "transform_args": {"unused": torch.tensor(1)}}
    result = train.train_loop(state, batch, accel, weights)
    base = sum(weights[k] * result[k] for k in weights)
    torch.testing.assert_close(base, result["codedrift/base_loss"])
    torch.testing.assert_close(result["loss"], base + result["codedrift/weighted_loss"])
    assert all(torch.isfinite(v).all() for v in result.values() if isinstance(v, torch.Tensor))
    assert not changed(before, model, "quantizer")
    assert changed(before, model, "encoder") == (scope in ("encoder", "both"))
    assert changed(before, model, "decoder") == (scope in ("decoder", "both"))
    assert any(not torch.equal(before_disc[k], v) for k, v in disc.state_dict().items())


@pytest.mark.parametrize("name,kind,weight", [
    ("base", "none", 0.), ("enc", "enc", 1.), ("proj", "proj", 10.),
    ("code", "code", 1.), ("code_paper_weight", "code", 100.)
])
def test_example_configs(name, kind, weight):
    import argbind
    from scripts import train
    args = argbind.load_args("conf/code_drift/" + name + ".yml")
    with argbind.scope(args):
        cfg = train.code_drift()
    assert cfg == CodeDriftConfig(True, kind, weight, "both")
    assert args["lambdas"] == {"mel/loss": 15., "adv/feat_loss": 2., "adv/gen_loss": 1.,
                               "vq/commitment_loss": .25, "vq/codebook_loss": 1.}
