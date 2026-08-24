from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

CLASS_NAMES = ["Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"]
NUM_CLASSES = 7
INPUT_DIM = 2
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1


class CanonicalSwaraTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_projection = nn.Linear(INPUT_DIM, D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS,
        )

        self.output_projection = nn.Linear(
            D_MODEL,
            NUM_CLASSES,
        )

    def forward(self, x):
        if x.ndim != 3:
            raise RuntimeError(
                f"Expected [B,T,C], got {tuple(x.shape)}"
            )
        if x.shape[-1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected input dimension {INPUT_DIM}, got {x.shape[-1]}"
            )

        x = self.input_projection(x)
        x = self.transformer(x)
        return self.output_projection(x)



def load_checkpoint(path: Path, device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model = CanonicalSwaraTransformer().to(device)

    state = checkpoint.get("model_state_dict")
    if state is None:
        raise RuntimeError("Checkpoint has no model_state_dict.")

    expected = set(model.state_dict())
    actual = set(state)

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint architecture mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    model.load_state_dict(state, strict=True)
    model.eval()

    return model, checkpoint


__all__ = [
    "CLASS_NAMES",
    "NUM_CLASSES",
    "CanonicalSwaraTransformer",
    "load_checkpoint",
]