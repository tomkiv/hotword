package engine

import (
	"github.com/vitalii/hotword/pkg/features"
	"github.com/vitalii/hotword/pkg/model"
)

// Engine coordinates audio preprocessing and model inference.
type Engine struct {
	model           model.Model
	sampleRate      int
	windowSize      int
	hopSize         int
	numMelFilters   int
	windowBuffer    []float32
	smoothProb      float32
	consecutiveHigh int // Count of consecutive frames above threshold
	samplesIngested int // Track how many samples have been ingested (for warmup)
}

// NewEngine creates a new inference engine.
func NewEngine(m model.Model, sampleRate int) *Engine {
	return &Engine{
		model:         m,
		sampleRate:    sampleRate,
		windowSize:    512,
		hopSize:       256,
		numMelFilters: 40,
		windowBuffer:  make([]float32, sampleRate), // 1 second buffer
	}
}

// Reset clears the engine's state, resetting the probability smoother and buffer.
// Call this after a detection or when starting a new listening session.
// Buffer is initialized with low-level noise to prevent onset false positives
// (zeros transitioning to audio can look like a hotword).
func (e *Engine) Reset() {
	e.smoothProb = 0
	e.consecutiveHigh = 0
	e.samplesIngested = 0
	e.windowBuffer = make([]float32, e.sampleRate)
	// Fill with low-level noise to mimic ambient silence
	for i := range e.windowBuffer {
		// Low amplitude pseudo-random noise using a simple formula
		// This avoids importing math/rand in the hot path
		e.windowBuffer[i] = float32(((i*7919)%1000)-500) / 50000.0 // Range: ~-0.01 to +0.01
	}
}

// PushSamples updates the sliding window buffer without running inference.
// Use this during silence to maintain buffer continuity without affecting smoothProb.
func (e *Engine) PushSamples(samples []float32) {
	if len(samples) >= len(e.windowBuffer) {
		copy(e.windowBuffer, samples[len(samples)-len(e.windowBuffer):])
	} else {
		copy(e.windowBuffer, e.windowBuffer[len(samples):])
		copy(e.windowBuffer[len(e.windowBuffer)-len(samples):], samples)
	}
	e.samplesIngested += len(samples)
}

// ProcessSingle evaluates a complete audio sample and returns the raw probability.
// Use this for verification/testing where each sample is evaluated independently.
// This does NOT use smoothing - it's meant for single-shot evaluation.
func (e *Engine) ProcessSingle(samples []float32) float32 {
	// Fill buffer with the complete sample
	e.PushSamples(samples)

	// Audio Preprocessing (Log-Mel Spectrogram)
	input := features.Extract(e.windowBuffer, e.sampleRate, e.windowSize, e.hopSize, e.numMelFilters)
	if input == nil {
		return 0
	}

	// Inference - return raw probability
	output := e.model.Forward(input)
	return output.Data[0]
}

// DebugInfo contains detailed information about engine state for debugging
type DebugInfo struct {
	RawProb         float32
	SmoothProb      float32
	ConsecutiveHigh int
	SamplesIngested int
	WarmupComplete  bool
	Detected        bool
}

// ProcessDebug is like Process but returns detailed debug information
func (e *Engine) ProcessDebug(samples []float32, threshold float32) DebugInfo {
	// 1. Update sliding window buffer
	e.PushSamples(samples)

	// 2. Check warmup
	warmupComplete := e.samplesIngested >= e.sampleRate

	// 3. Audio Preprocessing
	input := features.Extract(e.windowBuffer, e.sampleRate, e.windowSize, e.hopSize, e.numMelFilters)
	if input == nil {
		return DebugInfo{}
	}

	// 4. Inference
	output := e.model.Forward(input)
	rawProb := output.Data[0]

	// 5. Probability Smoothing
	const alpha = 0.3
	const decayFactor = 0.5
	const requiredConsecutive = 5
	const highProbThreshold = float32(0.9)

	if rawProb < highProbThreshold {
		e.smoothProb = e.smoothProb * decayFactor
		e.consecutiveHigh = 0
	} else {
		e.smoothProb = alpha*rawProb + (1-alpha)*e.smoothProb
		e.consecutiveHigh++
	}

	detected := warmupComplete && e.consecutiveHigh >= requiredConsecutive && e.smoothProb >= threshold

	return DebugInfo{
		RawProb:         rawProb,
		SmoothProb:      e.smoothProb,
		ConsecutiveHigh: e.consecutiveHigh,
		SamplesIngested: e.samplesIngested,
		WarmupComplete:  warmupComplete,
		Detected:        detected,
	}
}

// Process handles a chunk of audio samples for streaming detection.
// Returns the smoothed probability and a boolean indicating detection.
// Uses warmup period and consecutive frame counting to reduce false positives.
func (e *Engine) Process(samples []float32, threshold float32) (float32, bool) {
	info := e.ProcessDebug(samples, threshold)
	return info.SmoothProb, info.Detected
}
