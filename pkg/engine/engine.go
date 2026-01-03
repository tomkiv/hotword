package engine

import (
	"github.com/vitalii/hotword/pkg/features"
	"github.com/vitalii/hotword/pkg/model"
)

// Engine coordinates audio preprocessing and model inference.
type Engine struct {
	model         model.Model
	sampleRate    int
	windowSize    int
	hopSize       int
	numMelFilters int
	windowBuffer  []float32
	smoothProb    float32
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

// Process handles a chunk of audio samples and returns the hotword probability
// and a boolean indicating if it exceeded the threshold.
func (e *Engine) Process(samples []float32, threshold float32) (float32, bool) {
	// 1. Update sliding window buffer
	if len(samples) >= len(e.windowBuffer) {
		copy(e.windowBuffer, samples[len(samples)-len(e.windowBuffer):])
	} else {
		copy(e.windowBuffer, e.windowBuffer[len(samples):])
		copy(e.windowBuffer[len(e.windowBuffer)-len(samples):], samples)
	}

	// 2. Audio Preprocessing (Log-Mel Spectrogram)
	input := features.Extract(e.windowBuffer, e.sampleRate, e.windowSize, e.hopSize, e.numMelFilters)
	if input == nil {
		return 0, false
	}

	// 3. Inference
	output := e.model.Forward(input)

	rawProb := output.Data[0]

	// 4. Probability Smoothing (Exponential Moving Average)
	// alpha determines how fast the smoothed probability updates.
	// Low alpha (e.g. 0.1) = slow update, high smoothing (good for noise rejection)
	// High alpha (e.g. 0.9) = fast update, low smoothing
	const alpha = 0.3
	e.smoothProb = alpha*rawProb + (1-alpha)*e.smoothProb

	return e.smoothProb, e.smoothProb >= threshold
}
