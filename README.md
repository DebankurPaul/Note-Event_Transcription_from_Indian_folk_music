Automatic Note-Event Transcription of Indian Folk Music

A tonic-aware automatic note-event transcription system for Indian folk music.

The system takes an audio recording as input and produces a sequence of
seven-Swara note events with:

Swara identity

Onset time

Offset time

Duration

Prediction confidence

The final inference pipeline combines CREPE-based F0 extraction,
automatic tonic estimation, tonic-relative pitch representation, a
frozen Transformer-based seven-Swara classifier, and temporal event
decoding.

Overview

Indian melodic music is performed relative to a recording-specific tonic
(Sa). Therefore, absolute frequency alone is not an appropriate common
reference across recordings performed at different tonic frequencies.

The proposed pipeline first estimates the tonic and then converts the
extracted F0 trajectory into a tonic-relative cents representation.

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

Features

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

Sa  Re  Ga  Ma  Pa  Dha  Ni

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

Source Files

File

Purpose

src/audio_frontend.py

Audio standardization, CREPE F0 extraction, F0 cleaning, tonic estimation, and tonic-relative feature generation

src/model.py

Frozen Transformer-based seven-Swara classifier

src/decoder.py

Converts frame-level Swara predictions into note events

src/transcribe.py

Main inference entry point connecting the complete pipeline

Installation

1. Clone the Repository

git clone https://github.com/DebankurPaul/Note-Event_Transcription_from_Indian_folk_music.git

Then enter the repository:

cd Note-Event_Transcription_from_Indian_folk_music

2. Create a Virtual Environment

Python 3.11 is recommended.

Windows

python -m venv .venv
.venv\Scripts\activate

Linux / macOS

python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

Performing Transcription

The final system requires only:

An input audio recording

The trained model checkpoint

The checkpoint should be located at:

checkpoints/best_model.pt

Basic Command

python src/transcribe.py --audio "path/to/your/audio.mp3" --checkpoint "checkpoints/best_model.pt"

Example

python src/transcribe.py --audio "my_folk_song.mp3" --checkpoint "checkpoints/best_model.pt"

The complete inference process is:

Audio
  ↓
Audio Standardization
  ↓
CREPE F0 Extraction
  ↓
F0 Cleaning + Voicing
  ↓
Automatic Tonic Estimation
  ↓
Tonic-Relative Cents
  ↓
225-Hz Feature Generation
  ↓
Transformer Swara Classification
  ↓
Temporal Decoding
  ↓
Note-Event Transcription

No tonic.txt, pitch.txt, or note-event annotation file is required
when transcribing a new recording.

Output

All generated transcription files are automatically saved in:

outputs/

For an input file:

my_folk_song.mp3

the system generates:

outputs/
├── my_folk_song_note_events.csv
├── my_folk_song_note_events.json
└── my_folk_song_transcription_metadata.json

The outputs/ directory is created automatically if it does not exist.

CSV Output

The CSV file contains one row for each predicted note event.

Fields

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

Example

event_id,swara_id,swara,onset,offset,duration,confidence
1,0,Sa,0.000,0.120,0.120,0.91
2,1,Re,0.120,0.245,0.125,0.87
3,2,Ga,0.245,0.391,0.146,0.94

JSON Output

The JSON file contains the same note-event information in structured
JSON format.

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

Example:

my_folk_song_transcription_metadata.json

CPU and GPU Inference

The system automatically uses CUDA when a compatible NVIDIA GPU is
available.

CPU

python src/transcribe.py --audio "song.mp3" --checkpoint "checkpoints/best_model.pt" --device cpu

GPU

python src/transcribe.py --audio "song.mp3" --checkpoint "checkpoints/best_model.pt" --device cuda:0

If --device is omitted, the system automatically selects CUDA when
available and otherwise uses the CPU.

Model Checkpoint

The inference system uses:

checkpoints/best_model.pt

This is the frozen trained seven-Swara Transformer checkpoint.

Training is not required to perform transcription.

Input Audio

The system supports common audio formats including:

.wav
.flac
.mp3
.m4a

The input audio is internally standardized before pitch extraction.

Tonic Estimation

The system automatically estimates the tonic from the pitch information
extracted from the input recording.

A separate tonic annotation is therefore not required.

The estimated tonic is used to transform absolute F0 into tonic-relative
cents:

Absolute F0
     ↓
Estimated Sa / Tonic
     ↓
Tonic-Relative Cents

This allows recordings performed at different absolute tonic frequencies
to be represented in a common pitch space.

Swara Classification

The Transformer receives the tonic-normalized pitch representation
together with voicing information.

For each valid frame, it predicts one of:

Sa
Re
Ga
Ma
Pa
Dha
Ni

The frame-level predictions are then passed to the temporal decoder.

Note-Event Formation

Frame-level Swara predictions are converted into temporally structured
note events.

Each final event is represented as:

(Swara, Onset, Offset, Duration, Confidence)

For example:

Sa   1.20 s   1.68 s   0.48 s
Re   1.69 s   1.94 s   0.25 s
Ga   1.96 s   2.41 s   0.45 s

This converts frame-level Swara classification into a discrete
note-event transcription.

Important Notes

No Ground-Truth Files Required for Inference

The final inference pipeline requires only the input audio and trained
model checkpoint.

The following are not required for inference:

tonic.txt
pitch.txt
note-event annotations

These annotations are relevant to dataset preparation and evaluation,
not normal inference on a new recording.

Seven-Swara Representation

The current classifier predicts seven canonical Swara classes:

Sa, Re, Ga, Ma, Pa, Dha, Ni

It is therefore a seven-Swara transcription system rather than a complete
representation of every pitch nuance in Indian music.

Limitations

The current system does not explicitly model the complete expressive
structure of Indian melodic performance.

In particular, it does not fully capture:

Detailed Gamaka structures

Continuous melodic ornamentation

All microtonal pitch variations

Complete Raga grammar

Complex polyphonic note structures

All performance-specific pitch transitions

These aspects provide important directions for future development.

Evaluation

The final classifier was evaluated on five independent recordings.

The final Phase-31B model achieved a best validation accuracy of:

63.83%

Across the five evaluated recordings, frame-level accuracy varied
approximately between:

60% – 72%

This variation demonstrates that performance depends on the
characteristics of the input recording.

Detailed frame-level metrics, confusion matrices, note-event evaluation,
and computational/resource measurements are reported in the associated
research work.

Research Context

This project focuses on automatic note-event transcription of Indian
folk music using:

Neural pitch extraction

Automatic tonic estimation

Tonic-relative pitch representation

Transformer-based Swara classification

Temporal note-event decoding

The objective is to transform an audio recording into a temporally
structured sequence of canonical Swara events.

Reproducibility

For reproducible inference, use:

The provided best_model.pt checkpoint

The provided requirements.txt

The provided inference source files

The same input audio

A compatible Python environment

The final inference pipeline does not require retraining the model.

Citation

If you use this implementation or build upon this work, please cite the
associated research paper.

The research paper contains the complete methodology, experimental setup,
results, limitations, and references to the prior work used in the
development of the system.

License

This project is licensed under the MIT License. See the LICENSE file
for details.
