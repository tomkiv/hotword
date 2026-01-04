# Plan: Performance Optimization (Multithreading & Parallelism)

## Phase 1: Internal Loop Parallelism [checkpoint: d14fa27]
- [x] Task: Parallelize Conv2D and MaxPool2D (e35db24)
  - [x] Update `pkg/model/layers.go` to use goroutines for the filter/channel loops
  - [x] Ensure thread-safety for output buffer writes
  - [x] Write a benchmark to measure speedup on a large tensor
- [x] Task: Parallelize Conv2DBackward and DenseBackward (e35db24)
  - [x] Use a similar approach for the backward pass loops
- [x] Task: Conductor - User Manual Verification 'Phase 1: Internal Loop Parallelism' (Protocol in workflow.md) (d14fa27)

## Phase 2: Data-Parallel Trainer [checkpoint: ec623cf]
- [x] Task: Implement `ParallelTrainer` in `pkg/train/parallel.go` (ec623cf)
  - [x] Implement dataset sharding logic
  - [x] Create worker loops that perform local training steps
- [x] Task: Implement Weight Averaging (ec623cf)
  - [x] Write a function to average parameters across multiple `SequentialModel` instances
  - [x] Integrate synchronization at the end of each epoch
- [x] Task: Conductor - User Manual Verification 'Phase 2: Data-Parallel Trainer' (Protocol in workflow.md) (ec623cf)

## Phase 3: Integration and Benchmarking
- [~] Task: Update CLI and Configuration
  - [ ] Add `threads` flag to `cmd/train.go` and `cmd/listen.go`
  - [ ] Update `Trainer` initialization to use `ParallelTrainer` if threads > 1
- [ ] Task: Performance Benchmarking
  - [ ] Run a full training session on a dummy 1000-sample dataset and report speedup vs sequential
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration and Benchmarking' (Protocol in workflow.md)