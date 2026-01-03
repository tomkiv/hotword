# Plan: Live Audio Capture

## Phase 1: ALSA Integration and Capture [checkpoint: d6b2d4a]
- [x] Task: Implement CGO/ALSA wrapper for audio capture (d90b621)
  - [x] Write tests for ALSA device opening and state management
  - [x] Implement `Capture` loop in `pkg/audio/capture`
- [x] Task: Implement Audio Normalization and Streaming (d90b621)
  - [x] Write tests for PCM-to-Float32 conversion accuracy
  - [x] Implement the Go channel delivery mechanism
- [x] Task: Conductor - User Manual Verification 'Phase 1: ALSA Integration and Capture' (Protocol in workflow.md) (d6b2d4a)

## Phase 2: User Feedback and Calibration
- [x] Task: Implement text-based VU Meter (2bb43fd)
  - [x] Create utility to calculate RMS and peak levels from audio chunks
  - [x] Implement ASCII progress bar visualization
- [x] Task: Create standalone audio capture tool (61ba51a)
  - [x] Create `scripts/audio/verify_capture.go` to test mic input and VU meter
- [ ] Task: Conductor - User Manual Verification 'Phase 2: User Feedback and Calibration' (Protocol in workflow.md)

## Phase 3: macOS Support
- [x] Task: Implement macOS command-line capture (89d531d)
  - [ ] Implement `capture_darwin.go` using `sox` or `ffmpeg`
  - [ ] Verify capture on macOS with the standalone script
- [ ] Task: Conductor - User Manual Verification 'Phase 3: macOS Support' (Protocol in workflow.md)