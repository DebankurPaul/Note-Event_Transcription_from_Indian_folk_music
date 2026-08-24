## Automatic Note-Event Transcription of Indian Folk Music

A tonic-aware automatic note-event transcription system for Indian folk music. The system takes an audio recording as input and produces a sequence of seven-Swara note events with their onset, offset, duration, and prediction confidence.

The final inference pipeline combines neural fundamental-frequency estimation using CREPE, automatic tonic estimation, tonic-relative pitch representation, a frozen Transformer-based seven-Swara classifier, and temporal event decoding.

## Overview

Indian melodic music is performed relative to a recording-specific tonic (Sa). Therefore, directly using absolute frequency as the input representation can make the same Swara appear at different frequency locations across recordings.

This system addresses this problem by automatically estimating the tonic and representing the extracted fundamental-frequency trajectory relative to that tonic.

Final Pipeline

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
Frame-Level Swara Predictions
     │
     ▼
Temporal Decoder
     │
     ▼
Final Note Events

## Features

The final inference system provides:

Automatic fundamental-frequency extraction using CREPE

Automatic tonic estimation

Tonic-relative pitch normalization

225-Hz feature representation

Seven-class Swara classification

Frame-level Swara prediction

Temporal note-event segmentation

Onset estimation

Offset estimation

Duration estimation

Prediction confidence

CSV and JSON transcription output

Seven Swara Classes

The classifier predicts seven canonical Swara classes:

Sa
Re
Ga
Ma
Pa
Dha
Ni

Repository Structure

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

## Source Files

src/audio_frontend.py

Performs the audio frontend processing:

Audio
  ↓
16-kHz audio
  ↓
CREPE F0 extraction
  ↓
F0 cleaning / voicing
  ↓
Octave-error correction
  ↓
Automatic tonic estimation
  ↓
Tonic-relative cents
  ↓
225-Hz [relative_cents, voiced] features

src/model.py

Contains the frozen Transformer-based seven-Swara classifier used during inference.

The model receives two features per frame:

[relative_cents, voiced]

and predicts:

Sa / Re / Ga / Ma / Pa / Dha / Ni

src/decoder.py

Converts frame-level Swara predictions into temporally structured note events using the final temporal decoding procedure.

src/transcribe.py

The main entry point for the complete transcription pipeline. It connects the audio frontend, frozen Swara classifier, and temporal decoder.

## Installation

1. Clone the Repository

git clone https://github.com/DebankurPaul/Note-Event_Transcription_from_Indian_folk_music.git
cd Note-Event_Transcription

Replace (https://github.com/DebankurPaul/Note-Event_Transcription_from_Indian_folk_music.git) with the URL of this repository.

2. Create a Virtual Environment

Python 3.11 is recommended.

Windows

python -m venv .venv

Activate the environment:

.venv\Scripts\activate

Linux / macOS

python3 -m venv .venv

Activate the environment:

source .venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

Performing Transcription

Once the dependencies are installed and the trained checkpoint is available at:

checkpoints/best_model.pt

you can directly transcribe an audio recording.

Basic Command

python src/transcribe.py --audio "path/to/your/audio.mp3" --checkpoint "checkpoints/best_model.pt"

Example

python src/transcribe.py --audio "data/my_folk_song.mp3" --checkpoint "checkpoints/best_model.pt"

The system automatically performs:

Audio loading
      ↓
Audio standardization
      ↓
CREPE F0 extraction
      ↓
Voicing estimation
      ↓
Automatic tonic estimation
      ↓
Tonic-relative pitch conversion
      ↓
225-Hz feature generation
      ↓
Transformer Swara classification
      ↓
Temporal decoding
      ↓
Note-event generation

No tonic.txt, pitch.txt, or note-event annotation file is required for inference.

Output

The transcription results are automatically saved in the project's outputs/ directory.

For an input file such as:

my_folk_song.mp3

the output will be:

outputs/
├── my_folk_song_note_events.csv
├── my_folk_song_note_events.json
└── my_folk_song_transcription_metadata.json

The outputs/ directory is created automatically if it does not already exist.

CSV Output

The CSV transcription contains the following fields:

event_id
swara_id
swara
onset
offset
duration
confidence

Example:

event_id,swara_id,swara,onset,offset,duration,confidence
1,0,Sa,0.000,0.120,0.120,0.91
2,1,Re,0.120,0.245,0.125,0.87
3,2,Ga,0.245,0.391,0.146,0.94

Output Fields

Field

Description

event_id

Sequential identifier of the predicted event

swara_id

Numeric identifier of the predicted Swara

swara

Predicted Swara name

onset

Event start time in seconds

offset

Event end time in seconds

duration

Event duration in seconds

confidence

Mean prediction confidence associated with the event

JSON Output

The JSON file contains the same note-event information in structured JSON format.

Example:

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

Transcription Metadata

The metadata file records information about the inference run, including:

Input audio

Model checkpoint

Device used

Estimated tonic frequency

Frontend feature shape

Number of predicted events

Frontend processing information

Example filename:

my_folk_song_transcription_metadata.json

Using CPU or GPU

The system automatically uses CUDA when a compatible NVIDIA GPU is available.

CPU

To explicitly use the CPU:

python src/transcribe.py --audio "song.mp3" --checkpoint "checkpoints/best_model.pt" --device cpu

GPU

To explicitly use the first CUDA GPU:

python src/transcribe.py --audio "song.mp3" --checkpoint "checkpoints/best_model.pt" --device cuda:0

If --device is not specified, the system automatically selects CUDA when available and otherwise uses the CPU.

Checkpoint

The inference system uses the frozen trained Swara classifier checkpoint:

checkpoints/best_model.pt

The checkpoint is used only for inference. Training is not required to perform transcription.

The checkpoint corresponds to the final seven-Swara Transformer classifier used by the inference pipeline.

Input Audio

The system is designed to process common audio formats, including:

.wav
.flac
.mp3
.m4a

The input audio is internally standardized before pitch extraction.

Tonic Estimation

A major component of the system is automatic tonic estimation.

The system does not require the user to provide a separate tonic annotation. Instead, the tonic is estimated directly from the pitch information extracted from the input recording.

The estimated tonic is then used to transform absolute F0 into a tonic-relative cents representation:

Absolute F0
    ↓
Estimated Sa / Tonic
    ↓
Tonic-relative cents

This allows recordings performed at different absolute tonic frequencies to be represented in a common pitch space.

Swara Classification

The Transformer receives the tonic-normalized pitch representation together with the voicing information.

For each valid frame, the classifier predicts one of the seven canonical Swaras:

Sa
Re
Ga
Ma
Pa
Dha
Ni

The frame-level predictions are then passed to the temporal decoder.

Note-Event Formation

Frame-level Swara predictions are not directly treated as final note events.

The temporal decoder processes the frame sequence to produce discrete events.

Each final event is represented as:

(Swara, Onset, Offset, Duration, Confidence)

For example:

Sa   1.20 s   1.68 s   0.48 s
Re   1.69 s   1.94 s   0.25 s
Ga   1.96 s   2.41 s   0.45 s

This converts the frame-level classification output into a form suitable for note-event transcription.

Important Notes

No Ground-Truth Files Required for Inference

The final inference pipeline requires only the input audio and trained model checkpoint.

The following files are not required:

tonic.txt
pitch.txt
note-event annotations

These types of annotations are used during dataset preparation and evaluation, not during normal inference on a new recording.

Seven-Swara Representation

The current classifier predicts seven canonical Swara classes:

Sa, Re, Ga, Ma, Pa, Dha, Ni

It is therefore a seven-Swara transcription system rather than a complete representation of every possible pitch nuance in Indian music.

Limitations

The current system does not explicitly model the complete expressive structure of Indian melodic performance.

In particular, the current representation does not fully capture:

Detailed Gamaka structures

Continuous melodic ornamentation

All microtonal pitch variations

Complete Raga grammar

Complex polyphonic note structures

All performance-specific pitch transitions

These aspects provide important directions for future development.

Evaluation

The final classifier was evaluated on five independent recordings.

The final Phase-31B model achieved a best validation accuracy of approximately:

63.83%

Across the five evaluated recordings, frame-level accuracy varied approximately between:

60% – 72%

This variation demonstrates that performance depends on the characteristics of the input recording.

Detailed frame-level metrics, confusion matrices, note-event evaluation, and computational/resource measurements are reported in the associated research work.

Research Context

This project focuses on automatic note-event transcription of Indian folk music using:

Neural pitch extraction

Automatic tonic estimation

Tonic-relative pitch representation

Transformer-based Swara classification

Temporal note-event decoding

The objective is to transform an audio recording into a temporally structured sequence of canonical Swara events.

Reproducibility

For reproducible inference, use:

The provided best_model.pt checkpoint

The provided requirements.txt

The provided inference source files

The same input audio

A compatible Python environment

The final inference pipeline does not require retraining the model.

Citation

If you use this implementation or build upon this work, please cite the associated research paper.

The research paper contains the complete methodology, experimental setup, results, limitations, and references to the prior work used in the development of the system.

License

See the LICENSE file for licensing information.
