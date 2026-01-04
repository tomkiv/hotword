# Hotword Detection in Go

A minimalist hotword detection system written in Go, specifically designed for hobbyists building DIY smart home or embedded voice assistants on small embedded Linux systems like Raspberry Pi.

## Key Features

- **No Heavy Dependencies:** Native Go implementation of neural networks (CNN/RNN), no TensorFlow/PyTorch required.
- **Custom Training:** Train on your own "Hey Computer" or "Jarvis" hotwords using WAV files.
- **Real-time Inference:** Efficient, low-latency detection with built-in Voice Activity Detection (VAD).
- **Robustness:** Dynamic audio augmentation (noise, shift, gain) during training.
- **Cross-Platform:** Builds with standard `go build`.

## Installation

### Prerequisites

- Go 1.21 or higher.
- Linux (ALSA development headers) or macOS (CoreAudio).
  - Ubuntu/Debian: `sudo apt-get install libasound2-dev`

### Build

```bash
git clone https://github.com/tomkiv/hotword.git
cd hotword
go build -o hotword .
```

## Usage

The application uses command-line subcommands. Examples below assume you have built the binary as `hotword`.

### 1. Prepare Data

Organize your training data (16kHz mono WAV files) into two folders:
```
data/train/
      ├── hotword/    # WAV files containing the hotword
      └── background/ # WAV files of silence, noise, or speech with NO hotword
```

Organize your validation data (16kHz mono WAV files) into two folders:
```
data/validate/
      ├── hotword/    # WAV files containing the hotword
      └── background/ # WAV files of silence, noise, or speech with NO hotword
```

### 2. Train a Model

Train a new model using your dataset.

```bash
./hotword train --data ./data/train --out my_model.bin --epochs 50
```

**Common Options:**
- `--onset`: Automatically crop input files to where speech starts (useful for unaligned recordings).
- `--augment-prob 0.5`: Apply noise/shift augmentation to 50% of training samples.
- `--stride 8000`: Extract overlapping windows (8000 samples = 0.5s stride) from long files.
- `--threads 4`: Use parallel training.

### 3. Verify Model

Check the accuracy of your model against a dataset.

```bash
./hotword verify --model my_model.bin --data ./data/validate
```

### 4. Real-time Listening

Run Real-time inference to detect the hotword via microphone.

```bash
./hotword listen --model my_model.bin --threshold 0.6
```

**Actions:**
You can trigger commands upon detection:

```bash
# Execute a shell command
./hotword listen --model my_model.bin --action "say 'Yes?'"

# Run a script
./hotword listen --model my_model.bin --script ./wake_up.sh
```

**VAD & Tuning:**
- `--min-power`: Threshold to ignore silence.
- `--vad-energy` / `--vad-zcr`: Tuning for Voice Activity Detection gate.

## Configuration

You can also use a `config.yaml` file instead of flags. See `config.yaml` in the root directory for an example.
