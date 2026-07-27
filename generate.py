from pathlib import Path
import pretty_midi
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
#
# Each note event is represented as a tuple of 4 raw values:
#   pitch       - MIDI pitch, 0-127 (already discrete, no quantization needed)
#   duration    - seconds the note is held (note.end - note.start)
#   velocity    - how hard the note is struck, 0-127
#   time_shift  - seconds since the previous note *started* (0 for the first
#                 note in a file). This is what encodes rhythm/spacing.

def find_midi_files(root_dir):
    root = Path(root_dir)

    midi_files = list(root.rglob("*.mid"))
    midi_files += list(root.rglob("*.midi"))

    return sorted(midi_files)


def extract_notes(midi_file):
    """
    Returns a list of (pitch, duration, velocity, time_shift) tuples for one
    MIDI file, in time order.
    """

    midi = pretty_midi.PrettyMIDI(str(midi_file))

    raw_notes = []

    for instrument in midi.instruments:

        if instrument.is_drum:
            continue

        for note in instrument.notes:
            raw_notes.append(note)

    # Sort all notes across all instruments by start time
    raw_notes.sort(key=lambda note: note.start)

    events = []
    prev_start = None

    for note in raw_notes:

        duration = note.end - note.start
        time_shift = 0.0 if prev_start is None else note.start - prev_start

        events.append((note.pitch, duration, note.velocity, time_shift))

        prev_start = note.start

    return events


def load_all_notes(midi_files):
    all_events = []

    for midi_file in tqdm(midi_files, desc="Loading MIDI files"):
        all_events.extend(extract_notes(midi_file))

    return all_events


# ---------------------------------------------------------------------------
# Quantization: turn continuous duration/velocity/time_shift into bin indices
# ---------------------------------------------------------------------------
#
# Bin edges are computed from quantiles of the actual data, so bins adapt to
# whatever range of durations/velocities/gaps appear in your dataset instead
# of guessing fixed cutoffs.

class FeatureQuantizer:

    def __init__(self, values, num_bins=32):
        values = np.asarray(values, dtype=np.float64)

        quantiles = np.linspace(0, 1, num_bins + 1)
        edges = np.quantile(values, quantiles)

        # Guard against duplicate edges (e.g. many identical velocities)
        edges = np.unique(edges)

        if len(edges) < 2:
            edges = np.array([values.min(), values.max() + 1e-6])

        self.edges = edges
        self.num_bins = len(edges) - 1

        # Representative value for each bin, used when decoding back to a
        # real number (bin midpoint)
        self.bin_centers = (edges[:-1] + edges[1:]) / 2.0

    def encode(self, value):
        # np.digitize gives 1..num_bins for values inside range; clip to
        # valid bin index range
        idx = np.digitize([value], self.edges[1:-1], right=False)[0]
        return int(np.clip(idx, 0, self.num_bins - 1))

    def decode(self, idx):
        idx = int(np.clip(idx, 0, self.num_bins - 1))
        return float(self.bin_centers[idx])


def build_vocabularies(events, num_duration_bins=32, num_velocity_bins=32, num_timeshift_bins=32):
    """
    events: list of (pitch, duration, velocity, time_shift) tuples

    Returns a dict of quantizers/vocab sizes needed to encode/decode every
    feature.
    """

    durations = [e[1] for e in events]
    velocities = [e[2] for e in events]
    time_shifts = [e[3] for e in events]

    duration_q = FeatureQuantizer(durations, num_duration_bins)
    velocity_q = FeatureQuantizer(velocities, num_velocity_bins)
    timeshift_q = FeatureQuantizer(time_shifts, num_timeshift_bins)

    vocab = {
        "pitch_size": 128,  # fixed MIDI pitch range, no quantizer needed
        "duration_q": duration_q,
        "velocity_q": velocity_q,
        "timeshift_q": timeshift_q,
        "duration_size": duration_q.num_bins,
        "velocity_size": velocity_q.num_bins,
        "timeshift_size": timeshift_q.num_bins,
    }

    return vocab


def encode_events(events, vocab):
    """
    Convert raw (pitch, duration, velocity, time_shift) tuples into
    (pitch_idx, duration_idx, velocity_idx, timeshift_idx) index tuples.
    """

    encoded = []

    for pitch, duration, velocity, time_shift in events:

        encoded.append((
            int(pitch),
            vocab["duration_q"].encode(duration),
            vocab["velocity_q"].encode(velocity),
            vocab["timeshift_q"].encode(time_shift),
        ))

    return encoded


# ---------------------------------------------------------------------------
# Sequences / Dataset
# ---------------------------------------------------------------------------

def create_sequences(encoded_events, sequence_length=50):
    """
    Build input/target pairs. Each input is a window of `sequence_length`
    encoded event-tuples; each target is the single next event-tuple.
    """

    inputs = []
    targets = []

    for i in range(len(encoded_events) - sequence_length):

        input_seq = encoded_events[i:i + sequence_length]
        target = encoded_events[i + sequence_length]

        inputs.append(input_seq)
        targets.append(target)

    return inputs, targets


class MusicDataset(Dataset):
    """
    inputs:  list of sequences, each a list of (pitch, duration, velocity, time_shift) idx tuples
    targets: list of (pitch, duration, velocity, time_shift) idx tuples
    """

    def __init__(self, inputs, targets):
        inputs = np.array(inputs, dtype=np.int64)   # (N, seq_len, 4)
        targets = np.array(targets, dtype=np.int64)  # (N, 4)

        self.inputs = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


# ---------------------------------------------------------------------------
# Model: separate embeddings per feature -> concat -> LSTM -> separate heads
# ---------------------------------------------------------------------------

class MusicLSTM(nn.Module):

    def __init__(
        self,
        pitch_size,
        duration_size,
        velocity_size,
        timeshift_size,
        pitch_emb=128,
        duration_emb=32,
        velocity_emb=32,
        timeshift_emb=32,
        hidden_dim=256,
        num_layers=2
    ):
        super().__init__()

        self.pitch_embedding = nn.Embedding(pitch_size, pitch_emb)
        self.duration_embedding = nn.Embedding(duration_size, duration_emb)
        self.velocity_embedding = nn.Embedding(velocity_size, velocity_emb)
        self.timeshift_embedding = nn.Embedding(timeshift_size, timeshift_emb)

        combined_dim = pitch_emb + duration_emb + velocity_emb + timeshift_emb

        self.lstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # One output head per feature (multi-task prediction)
        self.fc_pitch = nn.Linear(hidden_dim, pitch_size)
        self.fc_duration = nn.Linear(hidden_dim, duration_size)
        self.fc_velocity = nn.Linear(hidden_dim, velocity_size)
        self.fc_timeshift = nn.Linear(hidden_dim, timeshift_size)

    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len, 4) long tensor - columns are
           [pitch_idx, duration_idx, velocity_idx, timeshift_idx]
        """

        pitch_idx = x[:, :, 0]
        duration_idx = x[:, :, 1]
        velocity_idx = x[:, :, 2]
        timeshift_idx = x[:, :, 3]

        pitch_vec = self.pitch_embedding(pitch_idx)
        duration_vec = self.duration_embedding(duration_idx)
        velocity_vec = self.velocity_embedding(velocity_idx)
        timeshift_vec = self.timeshift_embedding(timeshift_idx)

        combined = torch.cat(
            [pitch_vec, duration_vec, velocity_vec, timeshift_vec],
            dim=-1
        )

        output, hidden = self.lstm(combined, hidden)

        # Take only the last timestep
        output = output[:, -1, :]

        pitch_logits = self.fc_pitch(output)
        duration_logits = self.fc_duration(output)
        velocity_logits = self.fc_velocity(output)
        timeshift_logits = self.fc_timeshift(output)

        return pitch_logits, duration_logits, velocity_logits, timeshift_logits, hidden


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model,
    loader,
    device,
    num_epochs=20,
    learning_rate=1e-3
):
    """
    Trains on all 4 features jointly. Total loss is the sum of the
    cross-entropy losses for pitch, duration, velocity, and time_shift.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):

        total_loss = 0.0

        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for x, y in progress:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pitch_logits, duration_logits, velocity_logits, timeshift_logits, _ = model(x)

            pitch_loss = criterion(pitch_logits, y[:, 0])
            duration_loss = criterion(duration_logits, y[:, 1])
            velocity_loss = criterion(velocity_logits, y[:, 2])
            timeshift_loss = criterion(timeshift_logits, y[:, 3])

            loss = pitch_loss + duration_loss + velocity_loss + timeshift_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - avg loss: {avg_loss:.4f}")

    return model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_events(
    model,
    seed_sequence,
    vocab,
    num_notes=200,
    sequence_length=50,
    temperature=1.0,
    device="cpu"
):
    """
    Autoregressively generate encoded (pitch_idx, duration_idx, velocity_idx,
    timeshift_idx) tuples, then decode them back into real
    (pitch, duration, velocity, time_shift) values.

    seed_sequence: list of encoded idx-tuples, length == sequence_length
    """

    model.eval()

    generated_idx = list(seed_sequence)

    with torch.no_grad():
        for _ in range(num_notes):

            window = generated_idx[-sequence_length:]

            x = torch.tensor([window], dtype=torch.long, device=device)

            pitch_logits, duration_logits, velocity_logits, timeshift_logits, _ = model(x)

            def sample(logits):
                probs = F.softmax(logits.squeeze(0) / temperature, dim=-1)
                return torch.multinomial(probs, num_samples=1).item()

            next_pitch = sample(pitch_logits)
            next_duration = sample(duration_logits)
            next_velocity = sample(velocity_logits)
            next_timeshift = sample(timeshift_logits)

            generated_idx.append((next_pitch, next_duration, next_velocity, next_timeshift))

    # Decode back into real values
    decoded_events = []

    for pitch_idx, duration_idx, velocity_idx, timeshift_idx in generated_idx:

        pitch = pitch_idx  # pitch has no quantizer, index == pitch
        duration = vocab["duration_q"].decode(duration_idx)
        velocity = vocab["velocity_q"].decode(velocity_idx)
        time_shift = vocab["timeshift_q"].decode(timeshift_idx)

        decoded_events.append((pitch, duration, velocity, time_shift))

    return decoded_events


# ---------------------------------------------------------------------------
# MIDI / audio export
# ---------------------------------------------------------------------------

def events_to_midi(
    events,
    output_path,
    instrument_program=0
):
    """
    events: list of (pitch, duration, velocity, time_shift) real-valued tuples

    Reconstructs absolute note start times by accumulating time_shift, using
    the actual predicted duration and velocity per note.
    """

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=instrument_program)

    current_time = 0.0

    for i, (pitch, duration, velocity, time_shift) in enumerate(events):

        if i == 0:
            start = 0.0
        else:
            start = current_time + max(time_shift, 0.0)

        end = start + max(duration, 0.01)  # avoid zero-length notes

        note = pretty_midi.Note(
            velocity=int(np.clip(velocity, 1, 127)),
            pitch=int(np.clip(pitch, 0, 127)),
            start=start,
            end=end
        )

        instrument.notes.append(note)
        current_time = start

    midi.instruments.append(instrument)
    midi.write(str(output_path))

    print(f"Saved generated MIDI to {output_path}")


def midi_to_mp3(
    midi_path,
    mp3_path,
    sample_rate=44100,
    soundfont_path=None
):
    """
    Render a MIDI file to audio and export as MP3.

    If soundfont_path is given (a .sf2 file), uses FluidSynth for realistic
    instrument sound. Otherwise falls back to pretty_midi's built-in sine-wave
    synthesizer (no extra downloads needed, but sounds simple/electronic).

    Requires ffmpeg installed and on your PATH (called directly via
    subprocess - no pydub dependency).
    """

    import subprocess

    midi = pretty_midi.PrettyMIDI(str(midi_path))

    if soundfont_path is not None:
        audio = midi.fluidsynth(fs=sample_rate, sf2_path=str(soundfont_path))
    else:
        audio = midi.synthesize(fs=sample_rate)

    audio = audio / np.max(np.abs(audio) + 1e-9)
    audio_int16 = (audio * 32767).astype(np.int16)

    wav_path = Path(mp3_path).with_suffix(".wav")
    wavfile.write(str(wav_path), sample_rate, audio_int16)

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-codec:a", "libmp3lame",
            "-qscale:a", "2",
            str(mp3_path)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed - is it installed and on PATH?\n{result.stderr}"
        )

    print(f"Saved MP3 to {mp3_path}")


if __name__ == "__main__":

    data_path = "data/maestro-v1.0.0-midi/maestro-v1.0.0"

    files = find_midi_files(data_path)
    sample_files = files[:50]
    all_events = load_all_notes(sample_files)

    print(f"Total notes: {len(all_events):,}")

    vocab = build_vocabularies(
        all_events,
        num_duration_bins=32,
        num_velocity_bins=32,
        num_timeshift_bins=32
    )

    print(f"Pitch vocab size: {vocab['pitch_size']}")
    print(f"Duration bins: {vocab['duration_size']}")
    print(f"Velocity bins: {vocab['velocity_size']}")
    print(f"Time-shift bins: {vocab['timeshift_size']}")

    encoded_events = encode_events(all_events, vocab)

    sequence_length = 50

    inputs, targets = create_sequences(
        encoded_events,
        sequence_length
    )

    print(f"Total training samples: {len(inputs):,}")

    dataset = MusicDataset(inputs, targets)

    batch_size = 64

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    for x, y in loader:
        print("Input batch:", x.shape)   # (batch, seq_len, 4)
        print("Target batch:", y.shape)  # (batch, 4)
        break

    model = MusicLSTM(
        pitch_size=vocab["pitch_size"],
        duration_size=vocab["duration_size"],
        velocity_size=vocab["velocity_size"],
        timeshift_size=vocab["timeshift_size"]
    )

    print(model)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    print(device)

    # ---- Train ----
    num_epochs = 20

    model = train_model(
        model,
        loader,
        device,
        num_epochs=num_epochs,
        learning_rate=1e-3
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "sequence_length": sequence_length,
        },
        "music_lstm.pt"
    )
    print("Saved model checkpoint to music_lstm.pt")

    # ---- Generate ----
    seed_sequence = inputs[0]

    generated_events = generate_events(
        model,
        seed_sequence,
        vocab,
        num_notes=200,
        sequence_length=sequence_length,
        temperature=1.0,
        device=device
    )

    print(f"\nGenerated {len(generated_events)} events")
    print(generated_events[:10])

    events_to_midi(
        generated_events,
        output_path="generated_music.mid"
    )

