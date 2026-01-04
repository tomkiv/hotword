# Plan: Recurrent Model Support (GRU/LSTM)

## Phase 1: Bug Fixes and Trainer Update [checkpoint: 70d6414]
- [x] Task: Update `Trainer` for copied parameters (70d6414)
  - [x] Modify `Trainer.TrainStep` to call `SetParams` after `SGDUpdate`
  - [x] Write a regression test using a mock layer that returns copies of its weights
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Bug Fixes and Trainer Update' (Protocol in workflow.md)

## Phase 2: LSTM Implementation [checkpoint: 075641c]
- [x] Task: Implement `LSTMLayer` in `pkg/model/lstm.go` (075641c)
  - [x] Define `LSTMLayer` struct and constructor
  - [x] Implement `Forward` pass (standard 4-gate LSTM)
  - [x] Implement `Backward` pass (BPTT)
  - [x] Write unit tests for `LSTMLayer` forward and backward passes
- [ ] Task: Conductor - User Manual Verification 'Phase 2: LSTM Implementation' (Protocol in workflow.md)

## Phase 3: Integration and Persistence
- [x] Task: Update model builder and persistence (40cea39)
  - [ ] Add `lstm` support to `BuildModelFromConfig`
  - [ ] Add `lstm` support to `SaveModel` and `LoadModel` (Version 2 format)
  - [ ] Write tests for saving/loading LSTM models
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration and Persistence' (Protocol in workflow.md)

## Phase 4: Stateful Inference Engine
- [~] Task: Implement stateful recurrent inference
  - [ ] Update `model.Model` and `Layer` interfaces to support stateful operations (optional or via type assertion)
  - [ ] Update `Engine` to maintain and pass hidden states for recurrent layers
  - [ ] Verify stateful detection with a real-time hotword test
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Stateful Inference Engine' (Protocol in workflow.md)