# Automatic Note-Event Transcription of Indian Folk Music

A tonic-aware automatic note-event transcription system for Indian folk music.

The system takes an audio recording as input and produces a sequence of seven-Swara note events containing:

- **Swara identity**
- **Onset time**
- **Offset time**
- **Duration**
- **Prediction confidence**

The final inference pipeline combines CREPE-based F0 extraction, automatic tonic estimation, tonic-relative pitch representation, a frozen Transformer-based seven-Swara classifier, and temporal event decoding.

---

## Overview

Indian melodic music is performed relative to a recording-specific tonic (`Sa`). Therefore, absolute frequency alone is not an appropriate common reference across recordings performed at different tonic frequencies.

The system first estimates the tonic and then converts the extracted F0 trajectory into a tonic-relative cents representation.

### Final System Architecture

```text
Input Audio
     │
     ▼
Audio Standardization
     │
     ▼
CREPE F0 Extraction
     │
     ▼
F0 Cleaning + Voicing
     │
     ▼
Automatic Tonic Estimation
     │
     ▼
Tonic-Relative Cents
     │
     ▼
225-Hz Feature Representation
     │
     ▼
Frozen Transformer
Seven-Swara Classifier
     │
     ▼
Frame-Level Swara Prediction
     │
     ▼
Phase-31J Temporal Decoder
     │
     ▼
Swara Note Events
(Onset, Offset, Duration, Confidence)
```

---

## Seven-Swara Representation

The classifier predicts seven canonical Swara classes:

| Swara | Class |
|---|---:|
| Sa | 0 |
| Re | 1 |
| Ga | 2 |
| Ma | 3 |
| Pa | 4 |
| Dha | 5 |
| Ni | 6 |

The current system is intentionally limited to these seven canonical Swara classes.

---

## Repository Structure

```text
Note-Event_Transcription/
│
├── src/
│   ├── audio_frontend.py
│   ├── model.py
│   ├── decoder.py
│   └── transcribe.py
│
├── checkpoints/
│   └── best_model.pt
│
├── outputs/
│   └── .gitkeep
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

### Main Files

| File | Description |
|---|---|
| `src/audio_frontend.py` | Audio preprocessing, CREPE F0 extraction, voicing, tonic estimation, and tonic-relative feature generation |
| `src/model.py` | Frozen Transformer-based seven-Swara classifier |
| `src/decoder.py` | Converts frame-level Swara predictions into note events |
| `src/transcribe.py` | Main entry point for end-to-end inference |
| `checkpoints/best_model.pt` | Trained model checkpoint |
| `outputs/` | Directory where generated transcription files are saved |

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/DebankurPaul/Note-Event_Transcription_from_Indian_folk_music.git
cd Note-Event_Transcription_from_Indian_folk_music
```

## 2. Create a Virtual Environment

Python **3.11** is recommended.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Transcription System

The final system requires:

1. An input audio recording
2. `checkpoints/best_model.pt`

### Basic Command

```bash
python src/transcribe.py --audio "path/to/audio.mp3" --checkpoint "checkpoints/best_model.pt"
```

### Example

```bash
python src/transcribe.py --audio "my_folk_song.mp3" --checkpoint "checkpoints/best_model.pt"
```

That single command performs the complete transcription pipeline:

```text
Audio
  ↓
Audio Standardization
  ↓
CREPE F0 Extraction
  ↓
Automatic Tonic Estimation
  ↓
Tonic-Relative Pitch
  ↓
225-Hz Feature Generation
  ↓
Transformer Swara Classification
  ↓
Temporal Decoding
  ↓
Final Note-Event Transcription
```

No `tonic.txt`, `pitch.txt`, or note-event annotation file is required for inference on a new recording.

---

# Output

All generated files are automatically saved in the project's `outputs/` directory.

For example, if the input is:

```text
my_folk_song.mp3
```

the system generates:

```text
outputs/
├── my_folk_song_note_events.csv
├── my_folk_song_note_events.json
└── my_folk_song_transcription_metadata.json
```

The output directory is created automatically if it does not already exist.

---

## CSV Output

The CSV file contains one row per predicted note event.

### Fields

| Field | Description |
|---|---|
| `event_id` | Sequential event identifier |
| `swara_id` | Numeric Swara class |
| `swara` | Predicted Swara |
| `onset` | Event start time in seconds |
| `offset` | Event end time in seconds |
| `duration` | Event duration in seconds |
| `confidence` | Prediction confidence associated with the event |

### Example

```csv
event_id,swara_id,swara,onset,offset,duration,confidence
1,0,Sa,0.000,0.120,0.120,0.91
2,1,Re,0.120,0.245,0.125,0.87
3,2,Ga,0.245,0.391,0.146,0.94
```

---

## JSON Output

The JSON file contains the predicted note events in structured format.

Example:

```json
[
  {
    "event_id": 1,
    "swara_id": 0,
    "swara": "Sa",
    "onset": 0.0,
    "offset": 0.12,
    "duration": 0.12,
    "confidence": 0.91
  }
]
```

---

## Metadata Output

The metadata file records information about the inference run, including:

- Input audio
- Model checkpoint
- Device used
- Estimated tonic frequency
- Feature shape
- Number of predicted events
- Frontend processing information

---

# CPU and GPU Inference

The system automatically uses CUDA when a compatible NVIDIA GPU is available.

### Use CPU

```bash
python src/transcribe.py --audio "song.mp3" --checkpoint "checkpoints/best_model.pt" --device cpu
```

### Use GPU

```bash
python src/transcribe.py --audio "song.mp3" --checkpoint "checkpoints/best_model.pt" --device cuda:0
```

If `--device` is omitted, the system automatically selects CUDA when available and otherwise uses the CPU.

---

# Input Audio

The system supports common audio formats including:

```text
.wav
.flac
.mp3
.m4a
```

The audio is internally standardized before pitch extraction.

---

# Automatic Tonic Estimation

A separate tonic annotation is not required.

The system estimates the tonic directly from the pitch information extracted from the input recording.

The estimated tonic is then used to normalize F0:

```text
Absolute F0
     │
     ▼
Estimated Tonic (Sa)
     │
     ▼
Tonic-Relative Cents
```

This allows recordings performed at different absolute tonic frequencies to be represented in a common pitch space.

---

# Frame-Level Swara Classification

The Transformer receives the tonic-normalized pitch representation together with voicing information.

For each valid frame, the classifier predicts one of:

```text
Sa
Re
Ga
Ma
Pa
Dha
Ni
```

These frame-level predictions are subsequently passed to the temporal decoder.

---

# Note-Event Formation

Frame-level Swara predictions are converted into discrete note events.

Each final event contains:

```text
Swara
Onset
Offset
Duration
Confidence
```

For example:

```text
Sa    1.20 s    1.68 s    0.48 s
Re    1.69 s    1.94 s    0.25 s
Ga    1.96 s    2.41 s    0.45 s
```

The result is a temporally structured Swara transcription rather than only a frame-by-frame label sequence.

---

# Model Checkpoint

The final inference system uses:

```text
checkpoints/best_model.pt
```

This is the frozen trained seven-Swara Transformer checkpoint.

**Training is not required to perform transcription.**

---

# Evaluation

The final classifier was evaluated on five independent recordings.

The final Phase-31B model achieved a best validation accuracy of:

**63.83%**

Across the five evaluated recordings, frame-level accuracy varied approximately between:

**60% and 72%**

The variation demonstrates that model performance depends on the characteristics of the input recording.

Detailed frame-level metrics, confusion matrices, note-event evaluation, and computational/resource measurements are reported in the associated research paper.

---

# Reproducibility

For reproducible inference, use:

- The provided `best_model.pt` checkpoint
- The provided `requirements.txt`
- The provided inference source files
- The same input audio
- A compatible Python environment

The final inference pipeline does not require retraining the model.

---

# Research Context

This project focuses on automatic note-event transcription of Indian folk music using:

- Neural pitch extraction
- Automatic tonic estimation
- Tonic-relative pitch representation
- Transformer-based Swara classification
- Temporal note-event decoding

The objective is to transform an audio recording into a temporally structured sequence of canonical Swara events.

---

# Citation

If you use this implementation or build upon this work, please cite the associated research paper.

The research paper contains the complete methodology, experimental setup, results, limitations, and references to the prior work used in the development of the system.

---

# License

This project is licensed under the MIT License. See the `LICENSE` file for details.
