package engine

import (
	"github.com/vitalii/hotword/pkg/audio"
	"github.com/vitalii/hotword/pkg/model"
)

// Engine coordinates audio preprocessing and model inference.
type Engine struct {
	weights       *model.Tensor
	bias          []float32
	sampleRate    int
	windowSize    int
	hopSize       int
	numMelFilters int
	filterbank    [][]float32
}

// NewEngine creates a new inference engine.
func NewEngine(weights *model.Tensor, bias []float32, sampleRate int) *Engine {
	windowSize := 512
	hopSize := 256
	numMelFilters := 40
	
	fb := audio.CreateMelFilterbank(numMelFilters, windowSize, sampleRate, 0, float64(sampleRate/2))
	
	return &Engine{
		weights:       weights,
		bias:          bias,
		sampleRate:    sampleRate,
		windowSize:    windowSize,
		hopSize:       hopSize,
		numMelFilters: numMelFilters,
		filterbank:    fb,
	}
}

// Process handles a chunk of audio samples and returns the hotword probability
// and a boolean indicating if it exceeded the threshold.
func (e *Engine) Process(samples []float32, threshold float32) (float32, bool) {
	// 1. Audio Preprocessing
	stft := audio.STFT(samples, e.windowSize, e.hopSize)
	if len(stft) == 0 {
		return 0, false
	}
	
	melSpec := audio.MelSpectrogram(stft, e.numMelFilters, e.windowSize, e.sampleRate, 0, float64(e.sampleRate/2))
	
	// Flatten melSpec into a single Tensor for the Dense layer
	// In a real scenario, this would involve a CNN first, but for the basic engine,
	// we'll assume the model is a Dense layer acting on the flattened spectrogram.
	numFrames := len(melSpec)
	inputSize := numFrames * e.numMelFilters
	input := model.NewTensor([]int{inputSize})
	
	for i := 0; i < numFrames; i++ {
		for j := 0; j < e.numMelFilters; j++ {
			input.Data[i*e.numMelFilters+j] = melSpec[i][j]
		}
	}

	// 2. Inference
	output := model.Dense(input, e.weights, e.bias)
	
	// Basic sigmoid-like activation is missing, but model is expected to output probability.
	// For now, we'll just return the raw output as probability.
	prob := output.Data[0]
	
	return prob, prob >= threshold
}
