package engine

import (
	"testing"
	"github.com/tomkiv/hotword/pkg/model"
)

func TestEngine(t *testing.T) {
	t.Run("Inference End-to-End", func(t *testing.T) {
		sampleRate := 16000
		windowSize := 512
		hopSize := 256
		numMelFilters := 40
		
		// 1 second of audio (16000 samples)
		numSamples := 16000
		samples := make([]float32, numSamples)
		for i := range samples {
			samples[i] = 0.5
		}

		// Calculate number of frames for STFT
		numFrames := (numSamples - windowSize) / hopSize + 1
		inputSize := numFrames * numMelFilters

		// Create a tiny model for testing
		weights := model.NewTensor([]int{1, inputSize})
		for i := range weights.Data {
			weights.Data[i] = 0.01
		}
		bias := []float32{0.0}
		
		m := model.NewSequentialModel(
			model.NewDenseLayer(weights, bias),
			model.NewSigmoidLayer(),
		)
		e := NewEngine(m, sampleRate)
		
		prob, detected := e.Process(samples, 0.5)
		
		if prob <= 0 {
			t.Errorf("Expected positive probability, got %f", prob)
		}
		
		// If detected is true or false depends on weights, but it should be consistent
		_ = detected
	})
}
