package features

import (
	"testing"
)

func TestExtract(t *testing.T) {
	sampleRate := 16000
	windowSize := 512
	hopSize := 256
	numMelFilters := 40
	
	// 1 second of audio
	samples := make([]float32, sampleRate)
	for i := range samples {
		samples[i] = 0.5
	}
	
	tensor := Extract(samples, sampleRate, windowSize, hopSize, numMelFilters)
	
	if tensor == nil {
		t.Fatal("Expected tensor, got nil")
	}
	
	// Calculate expected frames: (16000 - 512) / 256 + 1 = 61
	expectedFrames := (len(samples) - windowSize) / hopSize + 1
	expectedSize := expectedFrames * numMelFilters
	
	if len(tensor.Data) != expectedSize {
		t.Errorf("Expected data size %d, got %d", expectedSize, len(tensor.Data))
	}

	if tensor.Shape[0] != expectedSize {
		t.Errorf("Expected shape [ %d ], got %v", expectedSize, tensor.Shape)
	}
}
