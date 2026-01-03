# Specification: Live Audio Capture

## Overview
This track implements the "ear" of the hotword detection system. It provides the capability to capture real-time audio from the system's microphone and stream it into the detection pipeline. This is critical for moving from file-based verification to real-world usage.

## Functional Requirements
- **Native ALSA Capture:** Implement a minimalist wrapper for ALSA (Linux) to capture 16-bit PCM audio at 16kHz.
- **macOS Support:** Use a command-line pipe (e.g., `sox` or `ffmpeg`) to capture audio on macOS.
- **Asynchronous Streaming:** Deliver captured audio chunks as `[]float32` slices through a Go channel.
- **VU Meter Utility:** Provide a text-based visualizer that displays real-time audio levels (ASCII bar) and peak amplitudes.
- **Error Handling:** Gracefully handle cases where the audio device is busy or unavailable.

## Implementation Details
- **Package:** `pkg/audio/capture`
- **Method:** Direct CGO/ALSA interaction for low latency on Linux/ARM. Command-line pipe for macOS.
- **Normalization:** Convert 16-bit PCM `int16` values to `float32` range [-1.0, 1.0] before streaming.

## Acceptance Criteria
- A standalone script can capture audio and display a working VU meter in the terminal on both Linux and macOS.
- The capture loop can run continuously for 5+ minutes without memory leaks or buffer overflows.
- Audio chunks delivered via the channel are correctly sized for the `SlidingWindow` (e.g., 512 or 1024 samples).