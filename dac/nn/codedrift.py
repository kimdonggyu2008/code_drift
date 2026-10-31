"""Code Drift objectives on the official DAC forward path.

Paper: https://arxiv.org/abs/2410.11025v2, section IV.
No new parameters, replacement quantizer, or inference-time correction.
"""

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class CodeDriftConfig:
    enabled: bool = False
    loss: str = "none"
    weight: float = 0.0
    scope: str = "both"

    def __post_init__(self):
        if self.loss not in ("none", "enc", "proj", "code"):
            raise ValueError("Code Drift loss must be none, enc, proj, or code")
        if self.scope not in ("both", "encoder", "decoder"):
            raise ValueError("Code Drift scope must be both, encoder, or decoder")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Code Drift weight must be finite and nonnegative")
        if self.loss == "none" and self.weight != 0:
            raise ValueError("loss=none requires weight=0")
        if not self.enabled and (self.loss != "none" or self.weight != 0 or self.scope != "both"):
            raise ValueError("Set code_drift.enabled=true to configure fine-tuning")

    def configure(self, model):
        """Call before constructing the optimizer or wrapping the model in DDP."""
        if not self.enabled:
            return
        model.requires_grad_(False)
        if self.scope in ("both", "encoder"):
            model.encoder.requires_grad_(True)
        if self.scope in ("both", "decoder"):
            model.decoder.requires_grad_(True)
        # Freeze in_proj, embedding, and out_proj, not just the embeddings.
        model.quantizer.quantizer_dropout = 0.0
        self.train_mode(model)

    def train_mode(self, model):
        """Reapply after model.train(); freezing parameters must NOT detach inputs."""
        if self.enabled:
            model.quantizer.eval()
            if self.scope == "encoder":
                model.decoder.eval()
            elif self.scope == "decoder":
                model.encoder.eval()

    def forward_kwargs(self):
        # A zero-weight control has no second-pass graph or extra RNG draws.
        if self.enabled and self.loss != "none" and self.weight > 0:
            return {"code_drift_loss": self.loss}
        return {}

    def to_dict(self):
        return asdict(self)

    def check_resume(self, metadata):
        previous = metadata.get("code_drift")
        if previous is None and not self.enabled:
            return  # Original DAC checkpoints have no Code Drift metadata.
        if previous != self.to_dict():
            raise ValueError(
                "Resume Code Drift settings differ from the checkpoint. "
                "Keep loss/weight/scope fixed, or start a new run from weights."
            )


def channel_l2_mean(left, right):
    """Unsquared channel L2, then mean over batch and ALL padded time frames."""
    if left.ndim != 3 or left.shape != right.shape:
        raise ValueError("Code Drift representations must have identical [B, C, T] shapes")
    # Float32 reduction avoids half-precision norm overflow under autocast.
    return torch.linalg.vector_norm(left.float() - right.float(), dim=1).mean()


def representation_loss(kind, first_encoded, first_projected, first_selected,
                        second_encoded, second_projected):
    if kind == "enc":
        return channel_l2_mean(second_encoded, first_encoded)
    if kind == "proj":
        return channel_l2_mean(second_projected, first_projected)
    if kind == "code":
        return channel_l2_mean(second_projected, first_selected.detach())
    raise ValueError("An active Code Drift objective must be enc, proj, or code")


def forward_with_code_drift(model, audio_data, sample_rate, n_quantizers, kind):
    """Both encodings stay inside one DAC.forward call, including under DDP.

    Uses the original encoder, ResidualVectorQuantize (and its STE), and decoder.
    The second input is the cropped reconstructed waveform WITHOUT detach or
    a second augmentation/normalization. No second decoder call is needed.
    """
    if kind not in ("enc", "proj", "code"):
        raise ValueError("Unknown Code Drift objective")
    if model.quantizer.training or model.quantizer.quantizer_dropout != 0:
        raise ValueError("Code Drift requires quantizer.eval() and dropout=0")
    if any(p.requires_grad for p in model.quantizer.parameters()):
        raise ValueError("Code Drift requires the entire quantizer to be frozen")

    length = audio_data.shape[-1]
    encoded = model.encoder(model.preprocess(audio_data, sample_rate))
    quantized, codes, projected, commitment, codebook = model.quantizer(
        encoded, n_quantizers
    )
    reconstructed = model.decode(quantized)[..., :length]
    returned = model.encoder(model.preprocess(reconstructed, sample_rate))
    returned_projected = selected = None
    if kind in ("proj", "code"):
        # DAC returns concatenated per-stage pre-quantization residual projections.
        _, _, returned_projected, _, _ = model.quantizer(returned, n_quantizers)
    if kind == "code":
        selected = torch.cat([
            stage.decode_code(codes[:, i])
            for i, stage in enumerate(model.quantizer.quantizers[:codes.shape[1]])
        ], dim=1)

    auxiliary = representation_loss(
        kind, encoded, projected, selected, returned, returned_projected
    )
    return {
        "audio": reconstructed,
        "z": quantized,
        "codes": codes,
        "latents": projected,
        "vq/commitment_loss": commitment,
        "vq/codebook_loss": codebook,
        "codedrift/loss": auxiliary,
    }
