from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from audio_frontend import extract_features_from_audio
from model import load_checkpoint
from decoder import decode_events


WINDOW_SIZE = 512
BATCH_SIZE = 8


@torch.no_grad()
def predict_frames(model, features, device):
    """
    Run the frozen Phase-31B model on non-overlapping 512-frame windows.
    """
    n = len(features)

    if n < WINDOW_SIZE:
        raise RuntimeError(
            f"Audio produced only {n} frontend frames; "
            f"at least {WINDOW_SIZE} frames are required."
        )

    starts = np.arange(
        0,
        n - WINDOW_SIZE + 1,
        WINDOW_SIZE,
        dtype=np.int64,
    )

    windows = np.stack(
        [features[s:s + WINDOW_SIZE] for s in starts],
        axis=0,
    )

    predictions = np.full(n, -1, dtype=np.int64)
    confidence = np.zeros(n, dtype=np.float32)

    for i in range(0, len(windows), BATCH_SIZE):
        batch = torch.from_numpy(
            np.ascontiguousarray(windows[i:i + BATCH_SIZE])
        ).to(device)

        logits = model(batch)
        probs = torch.softmax(logits.float(), dim=-1)

        conf, pred = torch.max(probs, dim=-1)

        pred = pred.cpu().numpy()
        conf = conf.cpu().numpy()

        for j in range(len(pred)):
            start = int(starts[i + j])
            end = start + WINDOW_SIZE

            predictions[start:end] = pred[j]
            confidence[start:end] = conf[j]

    return predictions, confidence


def save_events(events, output_csv, output_json):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "event_id",
        "swara_id",
        "swara",
        "onset",
        "offset",
        "duration",
        "confidence",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(events)

    output_json.write_text(
        json.dumps(events, indent=2),
        encoding="utf-8",
    )


def transcribe(
    audio_path,
    checkpoint_path,
    output_dir,
    device_name=None,
):
    device = torch.device(
        device_name
        if device_name
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    print("=" * 72)
    print("AUTOMATIC NOTE-EVENT TRANSCRIPTION")
    print("=" * 72)

    print(f"Audio      : {audio_path}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {device}")
    print()

    # ---------------------------------------------------------------
    # 1. Audio → tonic-relative 225-Hz features
    # ---------------------------------------------------------------
    (
        features,
        frame_times,
        tonic_hz,
        frontend_metadata,
    ) = extract_features_from_audio(
        audio_path,
        device=str(device),
    )

    print(f"Estimated tonic : {tonic_hz:.3f} Hz")
    print(f"Feature shape   : {features.shape}")

    # ---------------------------------------------------------------
    # 2. Load frozen Phase-31B Swara classifier
    # ---------------------------------------------------------------
    model, _ = load_checkpoint(
        Path(checkpoint_path),
        device,
    )

    # ---------------------------------------------------------------
    # 3. Frame-level Swara prediction
    # ---------------------------------------------------------------
    labels, confidence = predict_frames(
        model,
        features,
        device,
    )

    # ---------------------------------------------------------------
    # 4. Temporal decoding → note events
    # ---------------------------------------------------------------
    events = decode_events(
        labels,
        confidence,
    )

    # ---------------------------------------------------------------
    # 5. Save final transcription
    # ---------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = Path(audio_path).stem

    output_csv = (
        output_dir
        / f"{stem}_note_events.csv"
    )

    output_json = (
        output_dir
        / f"{stem}_note_events.json"
    )

    metadata_path = (
        output_dir
        / f"{stem}_transcription_metadata.json"
    )

    save_events(
        events,
        output_csv,
        output_json,
    )

    metadata = {
        "audio": str(
            Path(audio_path).resolve()
        ),
        "checkpoint": str(
            Path(checkpoint_path).resolve()
        ),
        "device": str(device),
        "estimated_tonic_hz": tonic_hz,
        "feature_shape": list(
            features.shape
        ),
        "num_predicted_events": len(events),
        "frontend": frontend_metadata,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("TRANSCRIPTION COMPLETE")
    print("=" * 72)

    print(
        f"Events : {len(events):,}"
    )

    print(
        f"CSV    : {output_csv}"
    )

    print(
        f"JSON   : {output_json}"
    )

    return events


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an audio recording "
            "into Swara note events."
        )
    )

    parser.add_argument(
        "--audio",
        required=True,
        help=(
            "Input audio file "
            "(.wav/.flac/.mp3/.m4a)."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to the frozen Phase-31B "
            "best_model.pt checkpoint."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory override. "
            "Defaults to <project>/outputs."
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "cpu, cuda, or cuda:0. "
            "Default: automatic."
        ),
    )

    args = parser.parse_args()

    # Project root:
    # <project>/
    #     src/
    #         transcribe.py
    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    # Default:
    # <project>/outputs/
    if args.output_dir:
        output_dir = (
            Path(args.output_dir)
            .expanduser()
            .resolve()
        )
    else:
        output_dir = (
            project_root
            / "outputs"
        )

    transcribe(
        audio_path=(
            Path(args.audio)
            .expanduser()
            .resolve()
        ),
        checkpoint_path=(
            Path(args.checkpoint)
            .expanduser()
            .resolve()
        ),
        output_dir=output_dir,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()