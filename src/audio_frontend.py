"""
========================================================================
PHASE 20A — AUDIO FRONTEND & AUTOMATIC TONIC ESTIMATION
========================================================================

PURPOSE
-------
Convert a completely new raw audio recording into the same pitch-domain
representation used by the frozen Phase 5G system.

INPUT
-----
Only a raw audio file:

    .wav / .flac / .mp3 / .m4a

NO:
    tonic.txt
    pitch.txt
    note-event ground truth
    Saraga annotations

OUTPUT
------
    f0_crepe.npy
    periodicity_crepe.npy
    frame_times_crepe.npy
    f0_octave_corrected.npy
    octave_correction_mask.npy
    octave_correction.json
    f0_225hz.npy
    periodicity_225hz.npy
    voiced_225hz.npy
    relative_cents_225hz.npy
    frontend_features_225hz.npy
    tonic_candidates.csv
    tonic.json
    frontend_metadata.json

The final feature matrix is:

    [T, 2]

columns:

    0 -> relative_cents
    1 -> voiced

This is the pitch-domain representation expected by the frozen
Phase 5G model.

IMPORTANT
---------
This phase does NOT run Phase 5G or Phase 19.

It only establishes:

    RAW AUDIO
        ↓
    F0
        ↓
    voicing
        ↓
    automatic tonic
        ↓
    relative cents
        ↓
    Phase-5G-compatible 225-Hz features

========================================================================
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchcrepe

from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import resample_poly
import soundfile as sf


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "phase11_accuracy_improvement"
    / "phase20a_audio_frontend"
)


# ======================================================================
# CONFIGURATION
# ======================================================================

TARGET_SR = 16000

# Saraga pitch representation is approximately 225 frames/second.
TARGET_FRAME_RATE = 225.0

# CREPE analysis hop.
# 5 ms at 16 kHz.
CREPE_HOP = 80

CREPE_MODEL = "tiny"

# RTX 2050-safe starting point.
CREPE_BATCH_SIZE = 512

# Hindustani singing / folk-music range.
FMIN = 70.0
FMAX = 1000.0

# CREPE periodicity threshold.
VOICING_THRESHOLD = 0.30

# Stronger threshold used during tonic estimation.
TONIC_THRESHOLD = 0.45

# Plausible tonic range.
TONIC_MIN_HZ = 70.0
TONIC_MAX_HZ = 300.0

# Pitch histogram resolution.
HISTOGRAM_BINS = 1200

# Smooth histogram in cents.
HISTOGRAM_SIGMA = 8.0

# Minimum separation between tonic candidates.
TONIC_MIN_SEPARATION = 40

# Number of candidates saved.
NUM_TONIC_CANDIDATES = 10

MIN_VOICED_FRAMES = 100


# ======================================================================
# DEVICE
# ======================================================================

def get_device(device_string=None):

    if device_string is not None:
        device = torch.device(device_string)

        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA requested but CUDA is not available."
                )

        return device

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ======================================================================
# AUDIO LOADING
# ======================================================================

def load_audio(path: Path):

    try:
        audio, sr = sf.read(
            str(path),
            always_2d=False,
            dtype="float32",
        )

    except Exception as exc:

        # Fallback for formats such as MP3/M4A if librosa is installed.
        try:

            import librosa

            audio, sr = librosa.load(
                str(path),
                sr=None,
                mono=True,
            )

            audio = np.asarray(
                audio,
                dtype=np.float32,
            )

        except Exception as fallback_exc:

            raise RuntimeError(
                "\nCould not decode audio.\n\n"
                f"File: {path}\n\n"
                "soundfile failed with:\n"
                f"{exc}\n\n"
                "librosa fallback also failed with:\n"
                f"{fallback_exc}\n\n"
                "For MP3/M4A, install an ffmpeg-backed audio "
                "decoder or convert the recording to WAV."
            ) from fallback_exc

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    # Stereo -> mono.
    if audio.ndim == 2:

        audio = np.mean(
            audio,
            axis=1,
        )

    if audio.ndim != 1:

        raise RuntimeError(
            f"Unexpected audio shape: {audio.shape}"
        )

    if len(audio) == 0:

        raise RuntimeError(
            "Audio file contains zero samples."
        )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Prevent pathological amplitudes.
    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    if peak > 1.0:

        audio = (
            audio / peak
        )

    return audio, int(sr)


# ======================================================================
# RESAMPLING
# ======================================================================

def resample_audio(
    audio,
    original_sr,
    target_sr,
):

    if original_sr == target_sr:

        return audio.astype(
            np.float32,
            copy=False,
        )

    gcd = math.gcd(
        original_sr,
        target_sr,
    )

    up = target_sr // gcd
    down = original_sr // gcd

    output = resample_poly(
        audio,
        up,
        down,
    )

    return np.asarray(
        output,
        dtype=np.float32,
    )


# ======================================================================
# CREPE
# ======================================================================

def extract_crepe(
    audio,
    device,
):

    waveform = torch.from_numpy(
        audio
    ).float()

    waveform = waveform.unsqueeze(
        0
    )

    waveform = waveform.to(
        device
    )

    print()
    print("=" * 72)
    print("CREPE F0 EXTRACTION")
    print("=" * 72)

    print(
        f"Model       : {CREPE_MODEL}"
    )

    print(
        f"Fmin        : {FMIN:.1f} Hz"
    )

    print(
        f"Fmax        : {FMAX:.1f} Hz"
    )

    print(
        f"Hop         : {CREPE_HOP} samples"
    )

    print(
        f"Frame step  : "
        f"{CREPE_HOP / TARGET_SR * 1000:.3f} ms"
    )

    print(
        f"Device      : {device}"
    )

    print(
        f"Batch size  : {CREPE_BATCH_SIZE}"
    )

    with torch.inference_mode():

        pitch, periodicity = (
            torchcrepe.predict(
                waveform,
                TARGET_SR,
                CREPE_HOP,
                FMIN,
                FMAX,
                CREPE_MODEL,
                batch_size=CREPE_BATCH_SIZE,
                device=device,
                return_periodicity=True,
                decoder=torchcrepe.decode.viterbi,
            )
        )

    pitch = (
        pitch
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    periodicity = (
        periodicity
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    pitch = np.asarray(
        pitch,
        dtype=np.float64,
    )

    periodicity = np.asarray(
        periodicity,
        dtype=np.float64,
    )

    pitch[
        ~np.isfinite(pitch)
    ] = 0.0

    periodicity[
        ~np.isfinite(periodicity)
    ] = 0.0

    periodicity = np.clip(
        periodicity,
        0.0,
        1.0,
    )

    return pitch, periodicity


# ======================================================================
# F0 CLEANING
# ======================================================================

def clean_f0(
    pitch,
    periodicity,
):

    valid = (
        np.isfinite(pitch)
        & np.isfinite(periodicity)
        & (pitch >= FMIN)
        & (pitch <= FMAX)
    )

    voiced = (
        valid
        & (
            periodicity
            >= VOICING_THRESHOLD
        )
    )

    f0 = np.zeros_like(
        pitch,
        dtype=np.float64,
    )

    f0[voiced] = (
        pitch[voiced]
    )

    # Median smoothing only over the pitch trajectory.
    if np.sum(voiced) >= 3:

        indices = np.arange(
            len(f0)
        )

        voiced_indices = (
            indices[voiced]
        )

        interpolated = np.interp(
            indices,
            voiced_indices,
            f0[voiced],
        )

        smoothed = median_filter(
            interpolated,
            size=5,
            mode="nearest",
        )

        f0[voiced] = (
            smoothed[voiced]
        )

    return f0, voiced


# ======================================================================
# RESAMPLE CREPE OUTPUT TO 225 Hz
# ======================================================================

def interpolate_to_225hz(
    f0,
    periodicity,
    voiced,
):

    original_times = (
        np.arange(
            len(f0),
            dtype=np.float64,
        )
        * CREPE_HOP
        / TARGET_SR
    )

    if len(original_times) == 0:

        raise RuntimeError(
            "CREPE returned no frames."
        )

    duration = (
        original_times[-1]
        if len(original_times) > 1
        else 0.0
    )

    target_count = max(
        1,
        int(
            round(
                duration
                * TARGET_FRAME_RATE
            )
        )
        + 1,
    )

    target_times = (
        np.arange(
            target_count,
            dtype=np.float64,
        )
        / TARGET_FRAME_RATE
    )

    # Interpolate F0 through voiced points only.
    voiced_indices = np.where(
        voiced
    )[0]

    if len(voiced_indices) < 2:

        raise RuntimeError(
            "Not enough voiced frames to construct "
            "the 225-Hz pitch representation."
        )

    voiced_times = (
        original_times[
            voiced_indices
        ]
    )

    voiced_f0 = (
        f0[
            voiced_indices
        ]
    )

    f0_225 = np.interp(
        target_times,
        voiced_times,
        voiced_f0,
        left=0.0,
        right=0.0,
    )

    # Periodicity is interpolated across all frames.
    periodicity_225 = np.interp(
        target_times,
        original_times,
        periodicity,
        left=0.0,
        right=0.0,
    )

    # Explicitly reconstruct the voiced mask.
    voiced_225 = (
        periodicity_225
        >= VOICING_THRESHOLD
    )

    # Remove F0 where the target grid is unvoiced.
    f0_225[
        ~voiced_225
    ] = 0.0

    return (
        target_times,
        f0_225,
        periodicity_225,
        voiced_225,
    )


# ======================================================================
# TONIC HISTOGRAM
# ======================================================================

def build_tonic_histogram(
    f0,
    periodicity,
):

    valid = (
        np.isfinite(f0)
        & np.isfinite(periodicity)
        & (f0 >= FMIN)
        & (f0 <= FMAX)
        & (
            periodicity
            >= TONIC_THRESHOLD
        )
    )

    values = f0[
        valid
    ]

    weights = periodicity[
        valid
    ]

    if len(values) < MIN_VOICED_FRAMES:

        raise RuntimeError(
            "Too few reliable voiced frames for "
            "automatic tonic estimation.\n"
            f"Reliable frames: {len(values)}"
        )

    cents = (
        1200.0
        * np.log2(values)
    )

    pitch_class = np.mod(
        cents,
        1200.0,
    )

    histogram = np.zeros(
        HISTOGRAM_BINS,
        dtype=np.float64,
    )

    indices = (
        np.floor(
            pitch_class
        )
        .astype(np.int64)
        % 1200
    )

    np.add.at(
        histogram,
        indices,
        weights,
    )

    # Circular smoothing.
    extended = np.concatenate(
        [
            histogram,
            histogram,
            histogram,
        ]
    )

    smoothed = (
        gaussian_filter1d(
            extended,
            sigma=HISTOGRAM_SIGMA,
            mode="wrap",
        )
    )

    smoothed = smoothed[
        1200:2400
    ]

    total = (
        np.sum(smoothed)
    )

    if total > 0:

        smoothed /= total

    return (
        smoothed,
        values,
        weights,
    )


# ======================================================================
# PEAK DETECTION
# ======================================================================

def get_tonic_peaks(
    histogram,
):

    order = np.argsort(
        histogram
    )[::-1]

    selected = []

    for index in order:

        index = int(index)

        acceptable = True

        for previous in selected:

            distance = abs(
                index - previous
            )

            distance = min(
                distance,
                1200 - distance,
            )

            if (
                distance
                < TONIC_MIN_SEPARATION
            ):

                acceptable = False
                break

        if acceptable:

            selected.append(
                index
            )

        if len(selected) >= NUM_TONIC_CANDIDATES:

            break

    return selected


# ======================================================================
# OCTAVE SELECTION
# ======================================================================

def select_tonic_octave(
    pitch_class_cent,
    f0,
    periodicity,
):
    """
    Select the octave of a tonic pitch class.

    The important distinction is between:

        octave-normalized consistency
        and
        direct support for the actual tonic frequency.

    Example:

        139 Hz  -> possible Sa
        278 Hz  -> octave of Sa

    Both can have excellent octave-normalized consistency.
    Therefore direct F0 support and lower-register information
    are also used.
    """

    f0 = np.asarray(
        f0,
        dtype=np.float64,
    )

    periodicity = np.asarray(
        periodicity,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(f0)
        & np.isfinite(periodicity)
        & (f0 >= FMIN)
        & (f0 <= FMAX)
        & (
            periodicity
            >= TONIC_THRESHOLD
        )
    )

    values = f0[valid]
    weights = periodicity[valid]

    if len(values) == 0:
        return None, 0.0

    # --------------------------------------------------------------
    # BUILD PITCH-CLASS REFERENCE
    # --------------------------------------------------------------

    base = 2.0 ** (
        float(pitch_class_cent)
        / 1200.0
    )

    candidates = []

    for octave in range(-10, 11):

        tonic = (
            base
            * (2.0 ** octave)
        )

        if (
            TONIC_MIN_HZ
            <= tonic
            <= TONIC_MAX_HZ
        ):
            candidates.append(
                float(tonic)
            )

    if not candidates:
        return None, 0.0

    # --------------------------------------------------------------
    # WEIGHTED F0 QUANTILES
    # --------------------------------------------------------------

    order = np.argsort(values)

    sorted_values = values[order]
    sorted_weights = weights[order]

    cumulative = np.cumsum(
        sorted_weights
    )

    total_weight = cumulative[-1]

    if total_weight <= 0:
        return None, 0.0

    def weighted_quantile(q):

        target = (
            q * total_weight
        )

        idx = np.searchsorted(
            cumulative,
            target,
            side="left",
        )

        idx = min(
            max(int(idx), 0),
            len(sorted_values) - 1,
        )

        return float(
            sorted_values[idx]
        )

    q10 = weighted_quantile(
        0.10
    )

    q25 = weighted_quantile(
        0.25
    )

    q50 = weighted_quantile(
        0.50
    )

    rows = []

    # --------------------------------------------------------------
    # SCORE EACH OCTAVE
    # --------------------------------------------------------------

    for tonic in candidates:

        # ==========================================================
        # 1. DIRECT TONIC SUPPORT
        # ==========================================================
        #
        # This is the important part.
        #
        # For a 139-Hz tonic:
        #
        #     F0 ≈ 139 Hz -> strong support
        #
        # For a 278-Hz candidate:
        #
        #     F0 ≈ 139 Hz -> weak direct support
        #
        # even though 139 Hz is an octave-related frequency.
        # ==========================================================

        cents_from_tonic = (
            1200.0
            * np.log2(
                values / tonic
            )
        )

        direct_error = np.abs(
            cents_from_tonic
        )

        direct_kernel = np.exp(
            -0.5
            * (
                direct_error / 55.0
            ) ** 2
        )

        direct_support = float(
            np.sum(
                direct_kernel
                * weights
            )
            / (
                np.sum(weights)
                + 1e-12
            )
        )

        # ==========================================================
        # 2. LOWER-RANGE SUPPORT
        # ==========================================================

        log_distance_q25 = abs(
            np.log2(
                tonic / q25
            )
        )

        range_score = float(
            np.exp(
                -0.5
                * (
                    log_distance_q25
                    / 0.65
                ) ** 2
            )
        )

        # ==========================================================
        # 3. OCTAVE-NORMALIZED CONSISTENCY
        # ==========================================================

        log_ratio = np.log2(
            values / tonic
        )

        nearest_octave = np.round(
            log_ratio
        )

        cents_error = (
            1200.0
            * (
                log_ratio
                - nearest_octave
            )
        )

        octave_scores = np.exp(
            -0.5
            * (
                cents_error / 35.0
            ) ** 2
        )

        octave_consistency = float(
            np.sum(
                octave_scores
                * weights
            )
            / (
                np.sum(weights)
                + 1e-12
            )
        )

        # ==========================================================
        # 4. VERY LOW TONIC PENALTY
        # ==========================================================

        if tonic < q10 * 0.75:

            low_penalty = 0.65

        elif tonic < q10 * 0.90:

            low_penalty = 0.85

        else:

            low_penalty = 1.0

        # ==========================================================
        # 5. FINAL OCTAVE SCORE
        # ==========================================================
        #
        # Direct tonic support receives the largest weight.
        #
        # This is intentional:
        #
        #     direct F0 evidence
        #          >
        #     octave-normalized evidence
        #
        # because octave-normalized evidence alone cannot distinguish
        # 139 Hz from 278 Hz.
        # ==========================================================

        score = (
            0.60 * direct_support
            + 0.25 * octave_consistency
            + 0.15 * range_score
        )

        score *= low_penalty

        rows.append(
            {
                "tonic_hz": float(
                    tonic
                ),
                "direct_tonic_support": float(
                    direct_support
                ),
                "octave_consistency": float(
                    octave_consistency
                ),
                "range_score": float(
                    range_score
                ),
                "low_octave_penalty": float(
                    low_penalty
                ),
                "octave_score": float(
                    score
                ),
                "q10_f0_hz": float(
                    q10
                ),
                "q25_f0_hz": float(
                    q25
                ),
                "median_f0_hz": float(
                    q50
                ),
            }
        )

    candidate_df = pd.DataFrame(
        rows
    )

    candidate_df = (
        candidate_df
        .sort_values(
            "octave_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    best = candidate_df.iloc[0]

    best_tonic = float(
        best["tonic_hz"]
    )

    best_score = float(
        best["octave_score"]
    )

    return (
        best_tonic,
        best_score,
        candidate_df,
    )

def correct_crepe_octave_errors(
    f0_hz: np.ndarray,
    periodicity: np.ndarray,
    voiced: np.ndarray,
):
    """
    Detect and conservatively correct a dominant CREPE octave-doubling
    error.

    The raw CREPE F0 is NEVER modified in-place.

    Strategy
    --------
    1. Find the dominant voiced F0 cluster.
    2. Test whether a lower-octave counterpart exists:
           upper_f0 / 2
    3. Require actual lower-frequency evidence.
    4. Correct only frames belonging to the dominant upper octave.
    5. Preserve all other F0 values.

    No tonic.txt, pitch.txt, or note-event ground truth is used.
    """

    f0 = np.asarray(
        f0_hz,
        dtype=np.float64,
    ).copy()

    periodicity = np.asarray(
        periodicity,
        dtype=np.float64,
    )

    voiced = np.asarray(
        voiced,
    )

    if not (
        len(f0)
        == len(periodicity)
        == len(voiced)
    ):
        raise ValueError(
            "f0_hz, periodicity and voiced must "
            "have identical lengths."
        )

    voiced_numeric = voiced.astype(
        np.float64
    )

    valid = (
        np.isfinite(f0)
        & np.isfinite(periodicity)
        & np.isfinite(voiced_numeric)
        & (voiced_numeric > 0.0)
        & (f0 >= FMIN)
        & (f0 <= FMAX)
        & (periodicity >= TONIC_THRESHOLD)
    )

    valid_values = f0[
        valid
    ]

    if len(valid_values) < 100:
        return (
            f0,
            np.zeros(
                len(f0),
                dtype=bool,
            ),
            {
                "correction_applied": False,
                "reason": "insufficient_valid_f0",
            },
        )

    # ==============================================================
    # STEP A — FIND DOMINANT F0 REGION
    # ==============================================================

    BIN_WIDTH_HZ = 5.0

    hist_min = float(
        max(
            FMIN,
            70.0,
        )
    )

    hist_max = float(
        min(
            FMAX,
            max(
                500.0,
                np.percentile(
                    valid_values,
                    99.0,
                ),
            ),
        )
    )

    bins = np.arange(
        hist_min,
        hist_max + BIN_WIDTH_HZ,
        BIN_WIDTH_HZ,
    )

    histogram, edges = np.histogram(
        valid_values,
        bins=bins,
    )

    if histogram.size == 0:
        return (
            f0,
            np.zeros(
                len(f0),
                dtype=bool,
            ),
            {
                "correction_applied": False,
                "reason": "empty_f0_histogram",
            },
        )

    # Find dominant bin.
    dominant_index = int(
        np.argmax(histogram)
    )

    dominant_lower = float(
        edges[
            dominant_index
        ]
    )

    dominant_upper = float(
        edges[
            dominant_index + 1
        ]
    )

    dominant_center = (
        dominant_lower
        + dominant_upper
    ) / 2.0

    dominant_count = int(
        histogram[
            dominant_index
        ]
    )

    total_valid = len(
        valid_values
    )

    dominant_ratio = (
        dominant_count
        / total_valid
    )

    # ==============================================================
    # STEP B — LOWER OCTAVE
    # ==============================================================

    lower_center = (
        dominant_center / 2.0
    )

    # Search ±10 Hz around the expected lower octave.
    LOWER_TOLERANCE_HZ = 10.0

    lower_mask = (
        valid_values
        >= (
            lower_center
            - LOWER_TOLERANCE_HZ
        )
    ) & (
        valid_values
        <= (
            lower_center
            + LOWER_TOLERANCE_HZ
        )
    )

    lower_count = int(
        np.sum(
            lower_mask
        )
    )

    lower_ratio = (
        lower_count
        / total_valid
    )

    # ==============================================================
    # STEP C — REQUIRE A REAL OCTAVE PAIR
    # ==============================================================

    #
    # We require:
    #
    #   1. substantial upper cluster
    #   2. non-trivial lower evidence
    #   3. plausible octave relationship
    #
    # The lower evidence threshold is deliberately conservative.
    #

    MIN_DOMINANT_RATIO = 0.08
    MIN_LOWER_RATIO = 0.02

    octave_pair_valid = (
        dominant_ratio
        >= MIN_DOMINANT_RATIO
        and lower_ratio
        >= MIN_LOWER_RATIO
        and lower_center
        >= FMIN
    )

    if not octave_pair_valid:
        return (
            f0,
            np.zeros(
                len(f0),
                dtype=bool,
            ),
            {
                "correction_applied": False,
                "reason": "no_reliable_octave_pair",

                "dominant_upper_hz": (
                    dominant_center
                ),

                "lower_octave_hz": (
                    lower_center
                ),

                "dominant_ratio": (
                    dominant_ratio
                ),

                "lower_ratio": (
                    lower_ratio
                ),
            },
        )

    # ==============================================================
    # STEP D — CORRECT ONLY THE DOMINANT UPPER CLUSTER
    # ==============================================================

    #
    # Do NOT divide the complete F0 track.
    #
    # Only frames close to the detected upper cluster are corrected.
    #

    UPPER_TOLERANCE_HZ = 7.5

    correction_mask = (
        valid
        & (
            np.abs(
                f0
                - dominant_center
            )
            <= UPPER_TOLERANCE_HZ
        )
    )

    corrected_f0 = f0.copy()

    corrected_f0[
        correction_mask
    ] = (
        corrected_f0[
            correction_mask
        ]
        / 2.0
    )

    corrected_count = int(
        np.sum(
            correction_mask
        )
    )

    corrected_ratio = (
        corrected_count
        / total_valid
    )

    # ==============================================================
    # STEP E — DIAGNOSTICS
    # ==============================================================

    diagnostics = {
        "correction_applied": (
            corrected_count > 0
        ),

        "dominant_upper_hz": float(
            dominant_center
        ),

        "dominant_upper_bin_start_hz": (
            dominant_lower
        ),

        "dominant_upper_bin_end_hz": (
            dominant_upper
        ),

        "lower_octave_hz": float(
            lower_center
        ),

        "dominant_ratio": float(
            dominant_ratio
        ),

        "lower_ratio": float(
            lower_ratio
        ),

        "corrected_frames": int(
            corrected_count
        ),

        "corrected_ratio": float(
            corrected_ratio
        ),

        "upper_tolerance_hz": (
            UPPER_TOLERANCE_HZ
        ),

        "lower_search_tolerance_hz": (
            LOWER_TOLERANCE_HZ
        ),

        "method": (
            "dominant F0 cluster "
            "+ lower-octave evidence "
            "+ selective octave-halving"
        ),
    }

    return (
        corrected_f0,
        correction_mask,
        diagnostics,
    )
# ======================================================================
# TONIC ESTIMATION
# ======================================================================

def estimate_tonic(
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    periodicity: np.ndarray,
):
    """
    Estimate tonic using:

        1. F0 validity filtering
        2. octave-folded pitch-class histogram
        3. pitch-class peak detection
        4. octave-specific F0 support
        5. direct tonic support
        6. lower-register support
        7. octave-pair diagnostics

    IMPORTANT:
        No tonic.txt, pitch.txt, or note-event ground truth is used.

    All octave realizations of each pitch-class candidate are retained
    so that octave-related candidates such as:

        139 Hz <-> 278 Hz

    can be explicitly compared.
    """

    # ==================================================================
    # STEP A — INPUT NORMALIZATION
    # ==================================================================

    f0_hz = np.asarray(
        f0_hz,
        dtype=np.float64,
    )

    voiced = np.asarray(
        voiced
    )

    periodicity = np.asarray(
        periodicity,
        dtype=np.float64,
    )

    if not (
        len(f0_hz)
        == len(voiced)
        == len(periodicity)
    ):
        raise ValueError(
            "f0_hz, voiced, and periodicity must have "
            "the same length.\n"
            f"f0_hz={len(f0_hz)}, "
            f"voiced={len(voiced)}, "
            f"periodicity={len(periodicity)}"
        )

    voiced_numeric = voiced.astype(
        np.float64
    )

    voiced_bool = (
        np.isfinite(voiced_numeric)
        & (voiced_numeric > 0.0)
    )

    # ==================================================================
    # STEP B — VALID F0
    # ==================================================================

    valid = (
        np.isfinite(f0_hz)
        & voiced_bool
        & np.isfinite(periodicity)
        & (periodicity >= TONIC_THRESHOLD)
        & (f0_hz >= FMIN)
        & (f0_hz <= FMAX)
    )

    valid_f0 = f0_hz[
        valid
    ].astype(
        np.float64
    )

    if len(valid_f0) < MIN_VOICED_FRAMES:
        raise RuntimeError(
            "Insufficient voiced F0 data for tonic estimation.\n"
            f"Valid voiced frames: {len(valid_f0)}"
        )

    # ==================================================================
    # STEP C — BUILD OCTAVE-FOLDED PITCH-CLASS HISTOGRAM
    # ==================================================================

    histogram, _, _ = build_tonic_histogram(
        f0_hz,
        periodicity,
    )

    if histogram is None:
        raise RuntimeError(
            "Tonic histogram construction failed."
        )

    histogram = np.asarray(
        histogram,
        dtype=np.float64,
    )

    if histogram.size == 0:
        raise RuntimeError(
            "Empty tonic histogram."
        )

    # ==================================================================
    # STEP D — FIND PITCH-CLASS PEAKS
    # ==================================================================

    peak_indices = get_tonic_peaks(
        histogram
    )

    if not peak_indices:
        raise RuntimeError(
            "No usable tonic pitch-class peaks were detected."
        )

    peak_cents = [
        float(index)
        for index in peak_indices
    ]

    # ==================================================================
    # STEP E — EVALUATE ALL OCTAVE REALIZATIONS
    # ==================================================================

    rows = []

    histogram_max = max(
        float(
            np.max(histogram)
        ),
        1e-12,
    )

    for pitch_class_cent in peak_cents:

        (
            _best_tonic,
            _best_octave_score,
            octave_candidates,
        ) = select_tonic_octave(
            float(
                pitch_class_cent
            ),
            f0_hz,
            periodicity,
        )

        if (
            octave_candidates is None
            or len(octave_candidates) == 0
        ):
            continue

        histogram_index = (
            int(
                round(
                    pitch_class_cent
                )
            )
            % 1200
        )

        raw_histogram_score = float(
            histogram[
                histogram_index
            ]
        )

        normalized_histogram_score = (
            raw_histogram_score
            / histogram_max
        )

        # --------------------------------------------------------------
        # RETAIN EVERY OCTAVE CANDIDATE.
        #
        # Do NOT use only octave_candidates.iloc[0].
        # --------------------------------------------------------------

        for _, octave_row in (
            octave_candidates.iterrows()
        ):

            rows.append(
                {
                    "pitch_class_cents": float(
                        pitch_class_cent
                    ),

                    "tonic_hz": float(
                        octave_row[
                            "tonic_hz"
                        ]
                    ),

                    "histogram_score": float(
                        normalized_histogram_score
                    ),

                    "direct_tonic_support": float(
                        octave_row[
                            "direct_tonic_support"
                        ]
                    ),

                    "octave_consistency": float(
                        octave_row[
                            "octave_consistency"
                        ]
                    ),

                    "range_score": float(
                        octave_row[
                            "range_score"
                        ]
                    ),

                    "low_octave_penalty": float(
                        octave_row[
                            "low_octave_penalty"
                        ]
                    ),

                    "octave_score": float(
                        octave_row[
                            "octave_score"
                        ]
                    ),

                    "q10_f0_hz": float(
                        octave_row[
                            "q10_f0_hz"
                        ]
                    ),

                    "q25_f0_hz": float(
                        octave_row[
                            "q25_f0_hz"
                        ]
                    ),

                    "median_f0_hz": float(
                        octave_row[
                            "median_f0_hz"
                        ]
                    ),
                }
            )

    if not rows:
        raise RuntimeError(
            "No valid octave-specific tonic candidates survived."
        )

    # ==================================================================
    # STEP F — COMBINE PITCH-CLASS + OCTAVE EVIDENCE
    # ==================================================================
    #
    # select_tonic_octave() already computes:
    #
    #     octave_score =
    #         0.60 * direct_tonic_support
    #       + 0.25 * octave_consistency
    #       + 0.15 * range_score
    #
    # including its low-octave penalty.
    #
    # Therefore we combine:
    #
    #     histogram evidence
    #     +
    #     octave-specific evidence
    #
    # without reconstructing octave_score.
    # ==================================================================

    for row in rows:

        row["combined_score"] = (
            0.40
            * row["histogram_score"]
            + 0.60
            * row["octave_score"]
        )

    # ==================================================================
    # STEP G — OCTAVE-PAIR DIAGNOSTICS
    # ==================================================================
    #
    # Explicitly identify pairs such as:
    #
    #     139 Hz <-> 278 Hz
    #
    # We DO NOT blindly divide the tonic by two.
    #
    # This section only records evidence for later selection.
    # ==================================================================

    OCTAVE_RATIO_TOLERANCE = 0.035

    for row in rows:

        tonic = float(
            row["tonic_hz"]
        )

        lower_octave = (
            tonic / 2.0
        )

        row["lower_octave_hz"] = float(
            lower_octave
        )

        row["octave_pair_detected"] = False

        row["octave_pair_lower_support"] = 0.0

        row["octave_pair_upper_support"] = float(
            row[
                "direct_tonic_support"
            ]
        )

        row["octave_pair_support_ratio"] = 0.0

        row["octave_pair_preference"] = 0.0

    # ------------------------------------------------------------------
    # Find corresponding lower-octave candidates.
    # ------------------------------------------------------------------

    for i, row in enumerate(rows):

        tonic = float(
            row["tonic_hz"]
        )

        lower_target = (
            tonic / 2.0
        )

        if (
            lower_target
            < TONIC_MIN_HZ
        ):
            continue

        best_index = None
        best_error = float(
            "inf"
        )

        for j, other in enumerate(rows):

            if i == j:
                continue

            other_tonic = float(
                other["tonic_hz"]
            )

            ratio_error = abs(
                (
                    other_tonic
                    / lower_target
                )
                - 1.0
            )

            if ratio_error < best_error:

                best_error = (
                    ratio_error
                )

                best_index = j

        if (
            best_index is None
            or best_error
            > OCTAVE_RATIO_TOLERANCE
        ):
            continue

        lower_row = rows[
            best_index
        ]

        lower_support = float(
            lower_row[
                "direct_tonic_support"
            ]
        )

        upper_support = float(
            row[
                "direct_tonic_support"
            ]
        )

        support_ratio = (
            lower_support
            / (
                upper_support
                + 1e-12
            )
        )

        row[
            "octave_pair_detected"
        ] = True

        row[
            "octave_pair_lower_support"
        ] = lower_support

        row[
            "octave_pair_upper_support"
        ] = upper_support

        row[
            "octave_pair_support_ratio"
        ] = support_ratio

    # ==================================================================
    # STEP H — CONSERVATIVE OCTAVE-PAIR PRIOR
    # ==================================================================
    #
    # We intentionally DO NOT force 278 -> 139.
    #
    # The pair information is diagnostic unless the lower octave has
    # meaningful direct support.
    #
    # This prevents an arbitrary "divide tonic by 2" hack.
    # ==================================================================

    OCTAVE_PAIR_MIN_LOWER_SUPPORT_RATIO = 0.25

    OCTAVE_PAIR_BONUS = 0.05

    for row in rows:

        if not row[
            "octave_pair_detected"
        ]:
            continue

        tonic = float(
            row["tonic_hz"]
        )

        lower_support = float(
            row[
                "octave_pair_lower_support"
            ]
        )

        upper_support = float(
            row[
                "octave_pair_upper_support"
            ]
        )

        support_ratio = float(
            row[
                "octave_pair_support_ratio"
            ]
        )

        # --------------------------------------------------------------
        # If this is the LOWER member of the octave pair and the
        # upper octave has meaningful support, give it a small bonus.
        # --------------------------------------------------------------

        if tonic < (
            row["lower_octave_hz"]
            * 1.01
        ):

            if (
                support_ratio
                >= 1.0
            ):

                row[
                    "octave_pair_preference"
                ] = (
                    OCTAVE_PAIR_BONUS
                )

        # --------------------------------------------------------------
        # If this is the UPPER member, only penalize it when the lower
        # octave has meaningful evidence.
        # --------------------------------------------------------------

        else:

            if (
                support_ratio
                >= OCTAVE_PAIR_MIN_LOWER_SUPPORT_RATIO
            ):

                row[
                    "octave_pair_preference"
                ] = (
                    -OCTAVE_PAIR_BONUS
                )

    # ------------------------------------------------------------------
    # Apply the conservative preference.
    # ------------------------------------------------------------------

    for row in rows:

        row[
            "combined_score_before_octave_pair"
        ] = float(
            row["combined_score"]
        )

        row[
            "combined_score"
        ] = (
            float(
                row["combined_score"]
            )
            + float(
                row[
                    "octave_pair_preference"
                ]
            )
        )

    # ==================================================================
    # STEP I — CREATE CANDIDATE DATAFRAME
    # ==================================================================

    candidates = pd.DataFrame(
        rows
    )

    candidates = (
        candidates
        .sort_values(
            "combined_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ==================================================================
    # STEP J — SELECT BEST TONIC
    # ==================================================================

    selected_tonic = float(
        candidates.iloc[0][
            "tonic_hz"
        ]
    )

    top_score = float(
        candidates.iloc[0][
            "combined_score"
        ]
    )

    if len(candidates) > 1:

        second_score = float(
            candidates.iloc[1][
                "combined_score"
            ]
        )

    else:

        second_score = 0.0

    # ==================================================================
    # STEP K — CONFIDENCE
    # ==================================================================

    confidence = (
        top_score
        - second_score
    ) / (
        top_score
        + 1e-12
    )

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )

    # ==================================================================
    # STEP L — RANK / SELECTED FLAGS
    # ==================================================================

    candidates.insert(
        0,
        "rank",
        np.arange(
            1,
            len(candidates) + 1,
        ),
    )

    candidates["selected"] = False

    candidates.loc[
        0,
        "selected",
    ] = True

    # ==================================================================
    # STEP M — DIAGNOSTICS
    # ==================================================================

    diagnostics = {
        "candidate_count": int(
            len(candidates)
        ),

        "best_score": float(
            top_score
        ),

        "second_best_score": float(
            second_score
        ),

        "confidence": float(
            confidence
        ),

        "voiced_frames": int(
            len(valid_f0)
        ),

        "f0_min_hz": float(
            np.min(valid_f0)
        ),

        "f0_max_hz": float(
            np.max(valid_f0)
        ),

        "f0_median_hz": float(
            np.median(valid_f0)
        ),

        "method": (
            "octave-folded pitch-class histogram "
            "+ direct tonic F0 support "
            "+ octave consistency "
            "+ lower-register support "
            "+ octave-pair diagnostic"
        ),
    }

    # ==================================================================
    # RETURN
    # ==================================================================

    return (
        selected_tonic,
        confidence,
        candidates,
        histogram,
    )
# ======================================================================
# RELATIVE CENTS
# ======================================================================

def make_relative_cents(
    f0,
    voiced,
    tonic_hz,
):

    if tonic_hz <= 0:

        raise RuntimeError(
            f"Invalid tonic: {tonic_hz}"
        )

    relative_cents = np.zeros(
        len(f0),
        dtype=np.float32,
    )

    valid = (
        voiced
        & np.isfinite(f0)
        & (f0 > 0)
    )

    relative_cents[
        valid
    ] = (
        1200.0
        * np.log2(
            f0[valid]
            / tonic_hz
        )
    ).astype(
        np.float32
    )

    relative_cents[
        ~np.isfinite(
            relative_cents
        )
    ] = 0.0

    voiced_float = (
        voiced.astype(
            np.float32
        )
    )

    return (
        relative_cents,
        voiced_float,
    )


# ======================================================================
# SAVE
# ======================================================================

def save_outputs(
    output_dir,
    input_path,
    original_sr,
    duration,
    f0_crepe,
    periodicity_crepe,
    crepe_times,
    f0_octave_corrected,
    octave_correction_mask,
    octave_correction_diagnostics,
    f0_225,
    periodicity_225,
    voiced_225,
    frame_times,
    relative_cents,
    voiced_float,
    tonic_hz,
    tonic_confidence,
    tonic_candidates,
    device,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frontend_features = (
        np.column_stack(
            [
                relative_cents,
                voiced_float,
            ]
        )
        .astype(
            np.float32
        )
    )

    np.save(
        output_dir
        / "f0_crepe.npy",
        f0_crepe.astype(
            np.float32
        ),
    )

    np.save(
        output_dir
        / "periodicity_crepe.npy",
        periodicity_crepe.astype(
            np.float32
        ),
    )

    np.save(
        output_dir
        / "frame_times_crepe.npy",
        crepe_times,
    )

    np.save(
        output_dir
        / "f0_octave_corrected.npy",
        f0_octave_corrected.astype(np.float32),
    )

    np.save(
        output_dir
        / "octave_correction_mask.npy",
        octave_correction_mask.astype(np.uint8),
    )

    with open(
        output_dir / "octave_correction.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            octave_correction_diagnostics,
            f,
            indent=2,
        )

    np.save(
        output_dir
        / "f0_225hz.npy",
        f0_225.astype(
            np.float32
        ),
    )

    np.save(
        output_dir
        / "periodicity_225hz.npy",
        periodicity_225.astype(
            np.float32
        ),
    )

    np.save(
        output_dir
        / "voiced_225hz.npy",
        voiced_float.astype(
            np.float32
        ),
    )

    np.save(
        output_dir
        / "frame_times.npy",
        frame_times,
    )

    np.save(
        output_dir
        / "relative_cents_225hz.npy",
        relative_cents.astype(
            np.float32
        ),
    )

    np.save(
        output_dir
        / "frontend_features_225hz.npy",
        frontend_features,
    )

    tonic_candidates.to_csv(
        output_dir
        / "tonic_candidates.csv",
        index=False,
    )

    voiced_f0 = f0_225[
        voiced_225
    ]

    metadata = {
        "phase": "20A",
        "description": (
            "Raw-audio F0 extraction and "
            "automatic tonic estimation."
        ),
        "input_audio": str(
            input_path
        ),
        "input_sampling_rate": int(
            original_sr
        ),
        "target_sampling_rate": TARGET_SR,
        "audio_duration_seconds": float(
            duration
        ),
        "crepe_model": CREPE_MODEL,
        "crepe_hop_samples": CREPE_HOP,
        "crepe_frame_rate_hz": (
            TARGET_SR
            / CREPE_HOP
        ),
        "target_feature_rate_hz": (
            TARGET_FRAME_RATE
        ),
        "f0_min_hz": FMIN,
        "f0_max_hz": FMAX,
        "voicing_threshold": (
            VOICING_THRESHOLD
        ),
        "tonic_threshold": (
            TONIC_THRESHOLD
        ),
        "estimated_tonic_hz": (
            tonic_hz
        ),
        "tonic_confidence_diagnostic": (
            tonic_confidence
        ),
        "crepe_frame_count": int(
            len(f0_crepe)
        ),
        "feature_frame_count": int(
            len(f0_225)
        ),
        "voiced_frames": int(
            np.sum(voiced_225)
        ),
        "voiced_ratio": float(
            np.mean(voiced_225)
        ),
        "f0_min_observed_hz": (
            float(
                np.min(voiced_f0)
            )
            if len(voiced_f0)
            else None
        ),
        "f0_max_observed_hz": (
            float(
                np.max(voiced_f0)
            )
            if len(voiced_f0)
            else None
        ),
        "f0_median_observed_hz": (
            float(
                np.median(voiced_f0)
            )
            if len(voiced_f0)
            else None
        ),
        "feature_shape": [
            int(
                frontend_features.shape[0]
            ),
            int(
                frontend_features.shape[1]
            ),
        ],
        "feature_columns": [
            "relative_cents",
            "voiced",
        ],
        "octave_correction": octave_correction_diagnostics,
        "octave_correction_frames": int(
            np.sum(octave_correction_mask)
        ),
        "uses_pitch_txt": False,
        "uses_tonic_txt": False,
        "uses_ground_truth": False,
        "training": False,
        "cache_rebuild": False,
        "device": str(device),
    }

    with open(
        output_dir
        / "tonic.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "estimated_tonic_hz": (
                    tonic_hz
                ),
                "confidence_diagnostic": (
                    tonic_confidence
                ),
                "method": (
                    "periodicity-weighted "
                    "1200-cent pitch-class "
                    "histogram plus octave "
                    "consistency scoring"
                ),
            },
            f,
            indent=2,
        )

    with open(
        output_dir
        / "frontend_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


# ======================================================================
# ARGUMENTS
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 20A raw-audio frontend."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input audio.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output directory.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "cpu / cuda / cuda:0. "
            "Default: automatic."
        ),
    )

    return parser.parse_args()


# ======================================================================
# MAIN
# ======================================================================

def main():

    args = parse_args()

    input_path = (
        Path(args.input)
        .expanduser()
        .resolve()
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input audio not found:\n"
            f"{input_path}"
        )

    device = get_device(
        args.device
    )

    if args.output:

        output_dir = (
            Path(args.output)
            .expanduser()
            .resolve()
        )

    else:

        output_dir = (
            OUTPUT_ROOT
            / input_path.stem
        )

    print("=" * 72)
    print(
        "PHASE 20A — AUDIO FRONTEND & "
        "AUTOMATIC TONIC ESTIMATION"
    )
    print("=" * 72)

    print()
    print("NO TRAINING")
    print("NO PITCH.TXT")
    print("NO TONIC.TXT")
    print("NO NOTE-EVENT GROUND TRUTH")
    print("NO PHASE 5G CHECKPOINT MODIFICATION")
    print("NO PHASE 19 CHECKPOINT MODIFICATION")

    print()
    print(
        f"Input  : {input_path}"
    )

    print(
        f"Output : {output_dir}"
    )

    print(
        f"Device : {device}"
    )

    # ------------------------------------------------------------------
    # STEP 1
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 1 — LOAD AUDIO")
    print("=" * 72)

    audio, original_sr = load_audio(
        input_path
    )

    original_duration = (
        len(audio)
        / original_sr
    )

    print(
        f"Original SR       : "
        f"{original_sr} Hz"
    )

    print(
        f"Channels          : mono"
    )

    print(
        f"Duration          : "
        f"{original_duration:.3f} s"
    )

    # ------------------------------------------------------------------
    # STEP 2
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 2 — STANDARDIZE AUDIO")
    print("=" * 72)

    audio = resample_audio(
        audio,
        original_sr,
        TARGET_SR,
    )

    duration = (
        len(audio)
        / TARGET_SR
    )

    print(
        f"Target SR         : "
        f"{TARGET_SR} Hz"
    )

    print(
        f"Samples           : "
        f"{len(audio):,}"
    )

    print(
        f"Duration          : "
        f"{duration:.3f} s"
    )

    # ------------------------------------------------------------------
    # STEP 3
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 3 — CREPE F0")
    print("=" * 72)

    f0_crepe, periodicity_crepe = (
        extract_crepe(
            audio,
            device,
        )
    )

    crepe_times = (
        np.arange(
            len(f0_crepe),
            dtype=np.float64,
        )
        * CREPE_HOP
        / TARGET_SR
    )

    print(
        f"CREPE frames     : "
        f"{len(f0_crepe):,}"
    )

    # ------------------------------------------------------------------
    # STEP 4
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 4 — VOICING / F0 CLEANING")
    print("=" * 72)

    f0_clean, voiced_clean = (
        clean_f0(
            f0_crepe,
            periodicity_crepe,
        )
    )

    print(
        f"Voiced frames    : "
        f"{np.sum(voiced_clean):,}"
    )

    print(
        f"Voiced ratio     : "
        f"{np.mean(voiced_clean):.4f}"
    )

    if (
        np.sum(voiced_clean)
        < MIN_VOICED_FRAMES
    ):

        raise RuntimeError(
            "Too few voiced frames."
        )

    # ------------------------------------------------------------------
    # STEP 4A — SELECTIVE CREPE OCTAVE-ERROR CORRECTION
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 4A — CREPE OCTAVE-ERROR DIAGNOSTIC / CORRECTION")
    print("=" * 72)

    (
        f0_corrected,
        octave_correction_mask,
        octave_correction_diagnostics,
    ) = correct_crepe_octave_errors(
        f0_clean,
        periodicity_crepe,
        voiced_clean,
    )

    print(
        f"Correction applied : "
        f"{octave_correction_diagnostics.get('correction_applied', False)}"
    )

    print(
        f"Dominant upper F0  : "
        f"{octave_correction_diagnostics.get('dominant_upper_hz', float('nan')):.3f} Hz"
    )

    print(
        f"Lower octave       : "
        f"{octave_correction_diagnostics.get('lower_octave_hz', float('nan')):.3f} Hz"
    )

    print(
        f"Upper ratio        : "
        f"{octave_correction_diagnostics.get('dominant_ratio', 0.0):.4f}"
    )

    print(
        f"Lower ratio        : "
        f"{octave_correction_diagnostics.get('lower_ratio', 0.0):.4f}"
    )

    print(
        f"Corrected frames   : "
        f"{octave_correction_diagnostics.get('corrected_frames', 0):,}"
    )

    print(
        f"Corrected ratio    : "
        f"{octave_correction_diagnostics.get('corrected_ratio', 0.0):.4f}"
    )

    # ------------------------------------------------------------------
    # STEP 5
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 5 — CONVERT TO 225-HZ GRID")
    print("=" * 72)

    (
        frame_times,
        f0_225,
        periodicity_225,
        voiced_225,
    ) = interpolate_to_225hz(
        f0_corrected,
        periodicity_crepe,
        voiced_clean,
    )

    print(
        f"225-Hz frames    : "
        f"{len(f0_225):,}"
    )

    print(
        f"Frame interval   : "
        f"{1.0 / TARGET_FRAME_RATE:.6f} s"
    )

    # ------------------------------------------------------------------
    # STEP 6
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 6 — AUTOMATIC TONIC")
    print("=" * 72)

    (
        tonic_hz,
        tonic_confidence,
        tonic_candidates,
        histogram,
    ) = estimate_tonic(
        f0_225,
        voiced_225,
        periodicity_225,
    )

    print()
    print(
        f"Estimated tonic  : "
        f"{tonic_hz:.3f} Hz"
    )

    print(
        f"Diagnostic conf. : "
        f"{tonic_confidence:.4f}"
    )

    print()
    print("Top tonic candidates:")

    print(
        tonic_candidates[
            [
                "rank",
                "tonic_hz",
                "histogram_score",
                "octave_consistency",
                "combined_score",
                "selected",
            ]
        ].head(10).to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # STEP 7
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 7 — RELATIVE CENTS")
    print("=" * 72)

    (
        relative_cents,
        voiced_float,
    ) = make_relative_cents(
        f0_225,
        voiced_225,
        tonic_hz,
    )

    frontend_features = (
        np.column_stack(
            [
                relative_cents,
                voiced_float,
            ]
        )
    )

    valid_cents = (
        relative_cents[
            voiced_225
        ]
    )

    print(
        f"Feature shape    : "
        f"{frontend_features.shape}"
    )

    print(
        "Feature columns   : "
        "[relative_cents, voiced]"
    )

    if len(valid_cents):

        print(
            f"Relative cents   : "
            f"{np.min(valid_cents):.2f}"
            f" → "
            f"{np.max(valid_cents):.2f}"
        )

    # ------------------------------------------------------------------
    # STEP 8
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 8 — SAVE")
    print("=" * 72)

    save_outputs(
        output_dir=output_dir,
        input_path=input_path,
        original_sr=original_sr,
        duration=duration,
        f0_crepe=f0_clean,
        periodicity_crepe=periodicity_crepe,
        crepe_times=crepe_times,
        f0_octave_corrected=f0_corrected,
        octave_correction_mask=octave_correction_mask,
        octave_correction_diagnostics=octave_correction_diagnostics,
        f0_225=f0_225,
        periodicity_225=periodicity_225,
        voiced_225=voiced_225,
        frame_times=frame_times,
        relative_cents=relative_cents,
        voiced_float=voiced_float,
        tonic_hz=tonic_hz,
        tonic_confidence=tonic_confidence,
        tonic_candidates=tonic_candidates,
        device=device,
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 20A COMPLETE")
    print("=" * 72)

    print()
    print(
        f"Estimated tonic : "
        f"{tonic_hz:.3f} Hz"
    )

    print(
        f"Voiced ratio    : "
        f"{np.mean(voiced_225):.4f}"
    )

    print(
        f"Feature shape   : "
        f"{frontend_features.shape}"
    )

    print()
    print("[SAVED]")

    print(
        output_dir
        / "f0_crepe.npy"
    )

    print(
        output_dir
        / "periodicity_crepe.npy"
    )

    print(
        output_dir
        / "f0_octave_corrected.npy"
    )

    print(
        output_dir
        / "octave_correction_mask.npy"
    )

    print(
        output_dir
        / "octave_correction.json"
    )

    print(
        output_dir
        / "f0_225hz.npy"
    )

    print(
        output_dir
        / "periodicity_225hz.npy"
    )

    print(
        output_dir
        / "voiced_225hz.npy"
    )

    print(
        output_dir
        / "relative_cents_225hz.npy"
    )

    print(
        output_dir
        / "frontend_features_225hz.npy"
    )

    print(
        output_dir
        / "tonic_candidates.csv"
    )

    print(
        output_dir
        / "tonic.json"
    )

    print(
        output_dir
        / "frontend_metadata.json"
    )


if __name__ == "__main__":
    main()

def extract_features_from_audio(input_path, device=None):
    """
    Production inference API.

    Returns:
        features: [T, 2] float32 array
            Column 0 = tonic-relative cents
            Column 1 = voiced flag
        frame_times: [T] float64 array
        tonic_hz: estimated tonic frequency
        metadata: dictionary with frontend diagnostics
    """
    input_path = Path(input_path).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found:\n{input_path}")

    device_obj = get_device(device)

    audio, original_sr = load_audio(input_path)
    audio = resample_audio(audio, original_sr, TARGET_SR)
    duration = len(audio) / TARGET_SR

    f0_crepe, periodicity_crepe = extract_crepe(audio, device_obj)

    crepe_times = (
        np.arange(len(f0_crepe), dtype=np.float64)
        * CREPE_HOP
        / TARGET_SR
    )

    f0_clean, voiced_clean = clean_f0(
        f0_crepe,
        periodicity_crepe,
    )

    if np.sum(voiced_clean) < MIN_VOICED_FRAMES:
        raise RuntimeError("Too few voiced frames for transcription.")

    (
        f0_corrected,
        octave_correction_mask,
        octave_correction_diagnostics,
    ) = correct_crepe_octave_errors(
        f0_clean,
        periodicity_crepe,
        voiced_clean,
    )

    (
        frame_times,
        f0_225,
        periodicity_225,
        voiced_225,
    ) = interpolate_to_225hz(
        f0_corrected,
        periodicity_crepe,
        voiced_clean,
    )

    (
        tonic_hz,
        tonic_confidence,
        tonic_candidates,
        histogram,
    ) = estimate_tonic(
        f0_225,
        voiced_225,
        periodicity_225,
    )

    relative_cents, voiced_float = make_relative_cents(
        f0_225,
        voiced_225,
        tonic_hz,
    )

    features = np.column_stack(
        [relative_cents, voiced_float]
    ).astype(np.float32)

    if features.ndim != 2 or features.shape[1] != 2:
        raise RuntimeError(
            f"Unexpected frontend feature shape: {features.shape}"
        )

    metadata = {
        "input_audio": str(input_path),
        "input_sampling_rate": int(original_sr),
        "target_sampling_rate": TARGET_SR,
        "audio_duration_seconds": float(duration),
        "crepe_model": CREPE_MODEL,
        "target_feature_rate_hz": TARGET_FRAME_RATE,
        "estimated_tonic_hz": float(tonic_hz),
        "tonic_confidence_diagnostic": float(tonic_confidence),
        "feature_shape": [int(features.shape[0]), int(features.shape[1])],
        "feature_columns": ["relative_cents", "voiced"],
        "voiced_ratio": float(np.mean(voiced_225)),
        "octave_correction_frames": int(np.sum(octave_correction_mask)),
        "device": str(device_obj),
    }

    return features, frame_times, float(tonic_hz), metadata