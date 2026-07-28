import time
from pathlib import Path
from datetime import datetime

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
        try:
            all_events.extend(extract_notes(midi_file))
        except Exception as e:
            # Some MIDI files in large datasets are malformed - skip rather
            # than crash a multi-hour run over one bad file
            print(f"Skipping {midi_file} due to error: {e}")

    return all_events


# ---------------------------------------------------------------------------
# Quantization: turn continuous duration/velocity/time_shift into bin indices
# ---------------------------------------------------------------------------
#
# Bin edges are computed from quantiles of the actual data, so bins adapt to
# whatever range of durations/velocities/gaps appear in your dataset instead
# of guessing fixed cutoffs.
#
# BUGFIX: with a large pooled dataset (1000+ files), it only takes ONE file
# with an unusually long pause (or a slightly malformed file with a stray
# late note) to make the raw maximum value huge - e.g. a 40-second gap.
# np.quantile's top edge is always the raw max, so the *center* of the last
# bin (the value used when DECODING back to seconds during generation) would
# then be something like (normal_high_value + 40s) / 2 ~= 20 seconds.
# If the model samples that bin even once during a 200-note generation, it
# inserts a ~20-second silent gap into the output - which is exactly the
# "20 seconds of music, rest is silence" symptom.
#
# Fix: clip extreme outliers out of the data before computing quantile edges,
# so every bin center stays in a realistic range no matter what one weird
# file in a 1000+ file dataset contains.

class FeatureQuantizer:

    def __init__(self, values, num_bins=32, clip_percentile=99.5):
        values = np.asarray(values, dtype=np.float64)

        if clip_percentile is not None and len(values) > 0:
            cap = np.percentile(values, clip_percentile)
            values = np.clip(values, values.min(), cap)

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
        # valid bin index range. Values above the (clipped) top edge simply
        # fall into the last bin, same as before - the difference is that
        # the last bin's center is now sane instead of outlier-driven.
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

    # Velocity is already bounded to 0-127 by the MIDI spec, so outliers
    # aren't a concern there. Duration and time_shift are unbounded seconds
    # values pooled across many files, so they get outlier clipping.
    duration_q = FeatureQuantizer(durations, num_duration_bins, clip_percentile=99.5)
    velocity_q = FeatureQuantizer(velocities, num_velocity_bins, clip_percentile=None)
    timeshift_q = FeatureQuantizer(time_shifts, num_timeshift_bins, clip_percentile=99.5)

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
#
# IMPORTANT - memory-safe design:
# The old approach built every single (sequence_length) window into a Python
# list up front, then converted the whole thing into one big NumPy array.
# That's fine for 20 files, but with the full dataset (potentially millions
# of notes) it would try to allocate an array of many terabytes and crash.
#
# Instead, MusicDataset now stores the full encoded event stream ONCE as a
# compact NumPy array, and slices out each window on-the-fly in __getitem__.
# Memory usage stays roughly constant no matter how large the dataset is.

class MusicDataset(Dataset):
    """
    encoded_events: list of (pitch_idx, duration_idx, velocity_idx, timeshift_idx)
                    tuples, IN ORDER, covering the whole dataset.
    sequence_length: length of each input window.
    """

    def __init__(self, encoded_events, sequence_length=50):
        # Single compact array for the whole dataset - this is the only
        # large allocation, and it's proportional to (num_notes, 4), not
        # (num_windows, sequence_length, 4).
        self.data = np.array(encoded_events, dtype=np.int64)  # (N, 4)
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.data) - self.sequence_length)

    def __getitem__(self, index):
        window = self.data[index: index + self.sequence_length]         # (seq_len, 4)
        target = self.data[index + self.sequence_length]                # (4,)

        x = torch.from_numpy(window.copy())
        y = torch.from_numpy(target.copy())

        return x, y


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
    learning_rate=1e-3,
    grad_clip_norm=5.0
):
    """
    Trains on all 4 features jointly. Total loss is the sum of the
    cross-entropy losses for pitch, duration, velocity, and time_shift.

    Uses automatic mixed precision (AMP) when training on a CUDA GPU, and
    gradient clipping to keep LSTM training stable on larger datasets.
    Per-feature losses are logged separately so you can see which feature
    (pitch/duration/velocity/timeshift) is learning well vs struggling.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()

    for epoch in range(num_epochs):

        totals = {"loss": 0.0, "pitch": 0.0, "duration": 0.0, "velocity": 0.0, "timeshift": 0.0}

        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for x, y in progress:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pitch_logits, duration_logits, velocity_logits, timeshift_logits, _ = model(x)

                pitch_loss = criterion(pitch_logits, y[:, 0])
                duration_loss = criterion(duration_logits, y[:, 1])
                velocity_loss = criterion(velocity_logits, y[:, 2])
                timeshift_loss = criterion(timeshift_logits, y[:, 3])

                loss = pitch_loss + duration_loss + velocity_loss + timeshift_loss

            scaler.scale(loss).backward()

            # Gradient clipping - unscale first so the clip threshold is
            # applied to the real gradient magnitudes, not the scaled ones
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            totals["loss"] += loss.item()
            totals["pitch"] += pitch_loss.item()
            totals["duration"] += duration_loss.item()
            totals["velocity"] += velocity_loss.item()
            totals["timeshift"] += timeshift_loss.item()

            progress.set_postfix(
                loss=loss.item(),
                pitch=pitch_loss.item(),
                dur=duration_loss.item(),
                vel=velocity_loss.item(),
                shift=timeshift_loss.item()
            )

        n = len(loader)
        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"avg loss: {totals['loss'] / n:.4f} "
            f"(pitch {totals['pitch'] / n:.4f}, "
            f"duration {totals['duration'] / n:.4f}, "
            f"velocity {totals['velocity'] / n:.4f}, "
            f"timeshift {totals['timeshift'] / n:.4f})"
        )

    return model


def benchmark_epoch_time(model, dataset, device, batch_size=64, num_batches=20):
    """
    Times a small number of training batches to estimate how long one full
    epoch (and the whole training run) will take, WITHOUT committing to the
    full run. Useful before scaling up from 20 files to the full dataset.
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=4
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()

    it = iter(loader)
    batches_timed = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pitch_logits, duration_logits, velocity_logits, timeshift_logits, _ = model(x)
            loss = (
                criterion(pitch_logits, y[:, 0])
                + criterion(duration_logits, y[:, 1])
                + criterion(velocity_logits, y[:, 2])
                + criterion(timeshift_logits, y[:, 3])
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batches_timed += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    if batches_timed == 0:
        print("Not enough data to benchmark.")
        return None

    seconds_per_batch = elapsed / batches_timed
    batches_per_epoch = len(loader)
    seconds_per_epoch = seconds_per_batch * batches_per_epoch

    print(
        f"\nBenchmark: {seconds_per_batch:.3f}s/batch on this dataset "
        f"({batches_per_epoch} batches/epoch)"
    )
    print(f"Estimated time per epoch: {seconds_per_epoch / 60:.1f} minutes")

    return seconds_per_epoch


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

    seed_sequence: list/array of encoded idx-tuples, length == sequence_length
    """

    model.eval()

    generated_idx = [tuple(int(v) for v in row) for row in seed_sequence]

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
    checkpoint_path = "music_lstm.pt"

    # Set this to True to force retraining even if a checkpoint already exists
    force_retrain = False

    # Set this to True to use every MIDI file found under data_path instead
    # of just the first 20 (only relevant when actually training).
    use_full_dataset = True

    # If True, times a handful of batches before committing to a full
    # training run and prints an estimated time per epoch / full run.
    run_benchmark_first = True

    sequence_length = 50
    batch_size = 64
    num_epochs = 20

    # How many files to scan when a checkpoint already exists and we only
    # need a short seed sequence to kick off generation - no need to rescan
    # the whole 1000+ file dataset just to generate.
    num_seed_files = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected - training will run on CPU (slower). "
              "If you have an NVIDIA GPU, make sure a CUDA-enabled build of "
              "PyTorch is installed.")

    checkpoint_exists = Path(checkpoint_path).exists() and not force_retrain

    # -----------------------------------------------------------------
    # Checkpoint already exists: load weights + vocab and SKIP the full
    # file scan / encode / train pipeline entirely. We only need a small
    # seed sequence to start generation, so we scan a handful of files
    # instead of all 1000+.
    # -----------------------------------------------------------------
    if checkpoint_exists:
        print(f"Found checkpoint at {checkpoint_path} - loading weights, "
              f"skipping the full MIDI scan/training pipeline.")

        # weights_only=False is required here (PyTorch >= 2.6 defaults to
        # True) because our checkpoint's "vocab" dict contains custom
        # FeatureQuantizer objects, not just tensors. This is safe as long
        # as the checkpoint file is one you saved yourself / trust.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # IMPORTANT: reuse the exact vocab/sequence_length the model was
        # trained with, not freshly rebuilt ones - bin edges must match
        # what the model's embeddings/output heads were trained on.
        vocab = checkpoint["vocab"]
        sequence_length = checkpoint["sequence_length"]

        model = MusicLSTM(
            pitch_size=vocab["pitch_size"],
            duration_size=vocab["duration_size"],
            velocity_size=vocab["velocity_size"],
            timeshift_size=vocab["timeshift_size"]
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model weights loaded successfully. Skipping training.")

        seed_files = find_midi_files(data_path)[:num_seed_files]
        print(f"Scanning {len(seed_files)} file(s) for a generation seed "
              f"(not the full dataset).")

        seed_events = load_all_notes(seed_files)
        seed_encoded = encode_events(seed_events, vocab)

        if len(seed_encoded) < sequence_length:
            raise RuntimeError(
                f"Only found {len(seed_encoded)} notes across {num_seed_files} "
                f"seed file(s), but need at least {sequence_length} for a "
                f"seed sequence. Increase num_seed_files."
            )

        seed_sequence = np.array(seed_encoded[:sequence_length], dtype=np.int64)

    # -----------------------------------------------------------------
    # No checkpoint (or force_retrain=True): run the full pipeline.
    # -----------------------------------------------------------------
    else:
        print("No checkpoint found (or force_retrain=True) - "
              "running the full pipeline: scan all files, build vocab, train.")

        files = find_midi_files(data_path)

        if not use_full_dataset:
            files = files[:20]

        print(f"Found {len(files):,} MIDI files - using {len(files):,} of them")

        all_events = load_all_notes(files)

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

        # NOTE: sequences are no longer pre-built into a giant list/array.
        # MusicDataset slices windows on-the-fly from the single encoded
        # stream, so this stays memory-safe even for millions of notes.
        dataset = MusicDataset(encoded_events, sequence_length)

        print(f"Total training samples: {len(dataset):,}")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
            num_workers=4
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
        ).to(device)

        print(model)

        if run_benchmark_first:
            print("\nRunning a quick benchmark before committing to full training...")
            est_seconds_per_epoch = benchmark_epoch_time(model, dataset, device, batch_size=batch_size)

            if est_seconds_per_epoch is not None:
                num_epochs_planned = num_epochs
                est_total_minutes = (est_seconds_per_epoch * num_epochs_planned) / 60
                print(
                    f"Estimated total training time for {num_epochs_planned} epochs: "
                    f"~{est_total_minutes:.1f} minutes (~{est_total_minutes / 60:.1f} hours)\n"
                )

            # Re-create the model fresh so the benchmark's optimizer steps
            # don't affect the real training run
            model = MusicLSTM(
                pitch_size=vocab["pitch_size"],
                duration_size=vocab["duration_size"],
                velocity_size=vocab["velocity_size"],
                timeshift_size=vocab["timeshift_size"]
            ).to(device)

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
            checkpoint_path
        )
        print(f"Saved model checkpoint to {checkpoint_path}")

        seed_sequence = dataset[0][0].numpy()  # first window, shape (seq_len, 4)

    # -----------------------------------------------------------------
    # Generate (common path, whether we just trained or just loaded)
    # -----------------------------------------------------------------
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

    # ---- Timestamped output files, so each run keeps its own take instead
    #      of overwriting the previous generated_music.mid/.mp3 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    midi_path = f"generated_music_{timestamp}.mid"
    mp3_path = f"generated_music_{timestamp}.mp3"

    events_to_midi(
        generated_events,
        output_path=midi_path
    )

    # ---- Export to MP3 ----
    midi_to_mp3(
        midi_path=midi_path,
        mp3_path=mp3_path
    )