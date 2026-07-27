# Music Generation using LSTMs

A deep learning project exploring sequence modeling for symbolic music generation using Long Short-Term Memory (LSTM) recurrent networks. The model learns pitch, timing, and temporal patterns from classical MIDI piano performances in the MAESTRO dataset to generate new musical sequences.

---

## Overview

Generating realistic music requires capturing long-term dependencies, rhythmic cadence, and structural harmony. This project approaches music generation as a sequential auto-regressive task:

* **Input Representation:** Extracts pitch, note duration, and step time from MIDI files.
* **Architecture:** Multi-layer LSTM network trained to predict the next note/event given a preceding sequence context.
* **Sampling & Generation:** Generates new MIDI files starting from a seed prompt or random initialization.

---

## Repository Structure

```text
music-rnn/
├── data/
│   └── maestro-v1.0.0-midi/
│       └── maestro-v1.0.0/
│           ├── 2004/
│           ├── 2006/
│           ├── 2008/
│           ├── 2009/
│           ├── 2011/
│           └── ...
└── generate.py

```

---

## Dataset

This project uses the **MAESTRO Dataset (v1.0.0)**, a collection of over 200 hours of virtuosic piano performances captured with fine pitch and timing details:

1. Download the [MAESTRO v1.0.0 MIDI dataset](https://magenta.tensorflow.org/datasets/maestro).
2. Extract the contents into `data/` as shown in the directory structure above.

---

## Getting Started

### Prerequisites

* Python 3.8+
* Recommended dependencies:
* `torch` (or `tensorflow`)
* `pretty_midi` or `mido` (for MIDI parsing)
* `numpy`
* `matplotlib`



Install dependencies:

```bash
pip install torch pretty_midi numpy

```

---

## Usage

### 1. Generating Music

Run `generate.py` to sample a new sequence from a trained checkpoint and export it as a standard `.mid` file:

```bash
python generate.py --output output.mid --length 500 --temperature 1.0

```

#### Common Arguments

* `--output`: Output file path for the generated MIDI file.
* `--length`: Number of note events to generate.
* `--temperature`: Sampling temperature (higher = more experimental, lower = safer/more repetitive).

---

## How It Works

1. **MIDI Parsing:** MIDI events are parsed into a numerical matrix where each event represents pitch, step time (delay between notes), and note duration.
2. **Sequential Windowing:** Training data is split into sliding sequence windows (e.g., 100 timesteps) mapped to the immediate next event.
3. **Training:** An LSTM processes these sequences, optimizing cross-entropy loss for pitch prediction and MSE loss for timing attributes.
4. **Sampling Loop:** The model predicts step $N+1$, appends it to the input window, and slides forward to generate the entire sequence iteratively.

---


## License

Distributed under the MIT License. See `LICENSE` for details.
