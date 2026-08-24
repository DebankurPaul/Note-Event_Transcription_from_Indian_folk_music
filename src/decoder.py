from __future__ import annotations

import numpy as np

CLASS_NAMES = ["Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"]
NUM_CLASSES = 7

FRAME_RATE = 225.0
DT = 1.0 / FRAME_RATE

# Final selected decoder configuration: D_80ms_40ms
FILTER_WINDOW = 5
MIN_DURATION_MS = 80.0
MERGE_GAP_MS = 40.0


def majority_filter(labels: np.ndarray, window: int = FILTER_WINDOW) -> np.ndarray:
    """Apply the selected majority filter to frame-level Swara labels."""
    labels = np.asarray(labels, dtype=np.int64)
    if window <= 1:
        return labels.copy()

    radius = window // 2
    padded = np.pad(labels, (radius, radius), mode="edge")
    out = labels.copy()

    for i in range(len(labels)):
        seg = padded[i:i + window]
        valid = seg[seg >= 0]

        if len(valid) == 0:
            out[i] = -1
            continue

        counts = np.bincount(valid, minlength=NUM_CLASSES)
        winner = int(np.argmax(counts))

        if counts[winner] > len(valid) / 2:
            out[i] = winner

    return out


def merge_short_gaps(labels: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Fill short unlabeled gaps only when both neighboring labels agree."""
    out = np.asarray(labels, dtype=np.int64).copy()
    i = 0
    n = len(out)

    while i < n:
        if out[i] >= 0:
            i += 1
            continue

        start = i
        while i < n and out[i] < 0:
            i += 1
        end = i

        if end - start > max_gap_frames:
            continue

        left = int(out[start - 1]) if start > 0 else -1
        right = int(out[end]) if end < n else -1

        if left >= 0 and left == right:
            out[start:end] = left

    return out


def remove_short_events(labels: np.ndarray, min_duration_frames: int) -> np.ndarray:
    """Remove short isolated events unless both neighbors share the label."""
    out = np.asarray(labels, dtype=np.int64).copy()
    n = len(out)
    i = 0

    while i < n:
        label = int(out[i])

        if label < 0:
            i += 1
            continue

        start = i
        while i < n and int(out[i]) == label:
            i += 1
        end = i

        if end - start >= min_duration_frames:
            continue

        left = int(out[start - 1]) if start > 0 else -1
        right = int(out[end]) if end < n else -1

        if left >= 0 and left == right:
            out[start:end] = left
        else:
            out[start:end] = -1

    return out


def labels_to_events(
    labels: np.ndarray,
    confidence: np.ndarray | None = None,
) -> list[dict]:
    """Convert frame labels into final Swara note events."""
    labels = np.asarray(labels, dtype=np.int64)

    if confidence is None:
        confidence = np.ones(len(labels), dtype=np.float32)
    else:
        confidence = np.asarray(confidence, dtype=np.float32)

    events = []
    start = None
    current = None

    def flush(end: int):
        nonlocal start, current

        if start is None:
            return

        if end <= start:
            start = None
            current = None
            return

        sid = int(current)
        event_conf = float(np.mean(confidence[start:end]))

        events.append(
            {
                "event_id": len(events) + 1,
                "swara_id": sid,
                "swara": CLASS_NAMES[sid],
                "onset": start * DT,
                "offset": end * DT,
                "duration": (end - start) * DT,
                "confidence": event_conf,
            }
        )

        start = None
        current = None

    for i, label in enumerate(labels):
        label = int(label)

        if label < 0 or label >= NUM_CLASSES:
            flush(i)
            continue

        if start is None:
            start = i
            current = label
        elif label != current:
            flush(i)
            start = i
            current = label

    flush(len(labels))
    return events


def decode_events(
    frame_labels: np.ndarray,
    frame_confidence: np.ndarray | None = None,
) -> list[dict]:
    """Apply the final temporal decoder and return note-event transcription."""
    frame_labels = np.asarray(frame_labels, dtype=np.int64)

    if frame_confidence is None:
        frame_confidence = np.ones(len(frame_labels), dtype=np.float32)

    min_frames = max(
        1,
        round(MIN_DURATION_MS / 1000.0 / DT),
    )
    gap_frames = max(
        0,
        round(MERGE_GAP_MS / 1000.0 / DT),
    )

    decoded = majority_filter(frame_labels, FILTER_WINDOW)
    decoded = merge_short_gaps(decoded, gap_frames)
    decoded = remove_short_events(decoded, min_frames)

    return labels_to_events(decoded, frame_confidence)


__all__ = ["decode_events"]