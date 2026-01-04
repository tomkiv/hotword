# Plan: Voice Activity Detection (VAD)

## Phase 1: Core VAD Utility [checkpoint: 0046043]
- [x] Task: Implement VAD logic in pkg/audio/vad.go (ca375f4)
  - [x] Implement `CalculateRMS(samples []float32) float32`
  - [x] Implement `CalculateZCR(samples []float32) float32`
  - [x] Create `VAD` struct with threshold and hangover state
- [x] Task: Write unit tests for VAD (ca375f4)
  - [x] Test VAD with silence, white noise, and speech-like samples
- [x] Task: Conductor - User Manual Verification 'Phase 1: Core VAD Utility' (Protocol in workflow.md) (0046043)

## Phase 2: Engine Integration [checkpoint: a8a961d]
- [x] Task: Integrate VAD into pkg/engine.Engine (a8a961d)
  - [x] Add `VAD` instance to the `Engine` struct
  - [x] Update `Engine.Process` to skip inference based on VAD output
  - [x] Implement the "hangover" timer logic
- [x] Task: Conductor - User Manual Verification 'Phase 2: Engine Integration' (Protocol in workflow.md) (a8a961d)

## Phase 3: Configuration and CLI
- [x] Task: Expose VAD parameters via Viper and CLI (7950773)
  - [ ] Add `vad_energy_threshold` and `vad_zcr_threshold` to `config.yaml`
  - [ ] Update `cmd/listen.go` to support VAD configuration flags
- [x] Task: Implement VAD status in CLI output (f33712f)
  - [ ] Add a `[VAD: INACTIVE]` indicator to the listen command status line
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Configuration and CLI' (Protocol in workflow.md)