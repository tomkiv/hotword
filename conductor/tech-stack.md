# Tech Stack - Hotword Detection in Go

## Core Language & Runtime
- **Language:** Go (Golang) 1.21+
- **Rationale:** Leverages modern features like structured logging (`slog`) and generics while maintaining high performance and small binary sizes.

## Machine Learning Engine
- **Implementation:** Custom Native Go Neural Network Engine.
- **Rationale:** Aligns with the "strict minimalism" goal by implementing CNN/RNN layers, activation functions, and backpropagation natively in Go using standard slices and the `math` package, avoiding heavy CGO dependencies or external ML frameworks.

## Audio Processing
- **Audio Capture:** OS-Native Drivers (e.g., ALSA for Linux).
- **Processing Utilities:** Native Go implementations for STFT, Mel-spectrograms, and hybrid Energy/ZCR Voice Activity Detection (VAD).
- **Rationale:** Maintains a low dependency footprint while providing optimized signal processing tools for embedded systems.

## Data Persistence & Models
- **Model Format:** Custom Binary Format.
- **Rationale:** Optimized for minimal storage and fast loading on resource-constrained hardware like the Raspberry Pi Zero.

## CLI & Configuration
- **CLI Framework:** Cobra & Viper.
- **Rationale:** Provides a professional and extensible CLI interface for managing training, inference, and configuration, while the core library remains dependency-free.

## Build & Deployment
- **Tooling:** Standard `go build` and `go test`.
- **Target OS:** Linux (ARM/ARM64 for Raspberry Pi).
