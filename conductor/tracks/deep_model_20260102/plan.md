# Plan: Deep Model Architecture (CNN)

## Phase 1: Core Model Refactor [checkpoint: c31fe05]
- [x] Task: Define generic Layer and Model structures (c31fe05)
  - [x] Implement `Layer` interface in `pkg/model` (Forward, Backward, Params)
  - [x] Create `SequentialModel` to manage a slice of layers
  - [x] Write unit tests for generic sequential forward pass
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Core Model Refactor' (Protocol in workflow.md)

## Phase 2: Configuration and Initialization
- [x] Task: Implement configuration-based model builder (865305b)
  - [ ] Add logic to parse layer definitions from `config.yaml`
  - [ ] Implement Xavier/Glorot initialization for Conv2D and Dense layers
  - [ ] Write tests for building complex models from YAML snippets
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Configuration and Initialization' (Protocol in workflow.md)

## Phase 3: Structured Model Persistence
- [x] Task: Update binary format for multi-layer support (df1662f)
  - [ ] Update `SaveModel` to include layer types and shapes in the header
  - [ ] Update `LoadModel` to dynamically reconstruct the model from the binary
  - [ ] Write tests for saving/loading multi-layer CNNs
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Structured Model Persistence' (Protocol in workflow.md)

## Phase 4: Training Pipeline Integration
- [x] Task: Update Trainer for multi-layer support (581024b)
  - [ ] Refactor `Trainer.TrainStep` to loop through layers for forward and backward passes
  - [ ] Verify convergence on a multi-layer CNN architecture
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Training Pipeline Integration' (Protocol in workflow.md)