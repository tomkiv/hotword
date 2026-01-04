package train

import (
	"testing"
)

func TestAugmentor(t *testing.T) {
	config := AugmentorConfig{
		AugmentProb:   1.0, // Always augment
		MaxNoiseRatio: 0.5,
		MaxShiftMs:    100,
		MaxGainScale:  0.2,
	}
	
	// Create some background noise for mixing
	noise := Sample{Audio: make([]float32, 16000), IsHotword: false}
	for i := range noise.Audio {
		noise.Audio[i] = 0.1
	}
	
	aug := NewAugmentor(config, []Sample{noise})
	
	hotword := []float32{0.5, 0.5, 0.5}
	augmented := aug.Augment(hotword)
	
	if len(augmented) != len(hotword) {
		t.Errorf("Expected length %d, got %d", len(hotword), len(augmented))
	}
	
	// Check that it's different (since prob is 1.0)
	different := false
	for i := range hotword {
		if hotword[i] != augmented[i] {
			different = true
			break
		}
	}
	if !different {
		t.Error("Expected augmented audio to be different from original")
	}
}
