package audio

import (
	"testing"
)

func TestCalculateRMS(t *testing.T) {
	t.Run("Silence", func(t *testing.T) {
		samples := []float32{0, 0, 0, 0}
		rms := CalculateRMS(samples)
		if rms != 0 {
			t.Errorf("Expected 0, got %f", rms)
		}
	})

	t.Run("Constant Signal", func(t *testing.T) {
		samples := []float32{0.5, 0.5, 0.5, 0.5}
		rms := CalculateRMS(samples)
		if rms != 0.5 {
			t.Errorf("Expected 0.5, got %f", rms)
		}
	})
}

func TestCalculateZCR(t *testing.T) {
	t.Run("Zero Crossings", func(t *testing.T) {
		// 1 crossing: -1 to 1
		samples := []float32{-1, 1, 1, 1}
		zcr := CalculateZCR(samples)
		var expected float32 = 1.0 / 3.0
		if zcr != expected {
			t.Errorf("Expected %f, got %f", expected, zcr)
		}
	})

	t.Run("Multiple Crossings", func(t *testing.T) {
		// 3 crossings: 1 to -1, -1 to 1, 1 to -1
		samples := []float32{1, -1, 1, -1}
		zcr := CalculateZCR(samples)
		var expected float32 = 3.0 / 3.0
		if zcr != expected {
			t.Errorf("Expected %f, got %f", expected, zcr)
		}
	})
}

func TestVADIsSpeech(t *testing.T) {
	t.Run("Silent Sample", func(t *testing.T) {
		vad := NewVAD(0.1, 0.5, 300)
		samples := make([]float32, 1600) // 100ms
		if vad.IsSpeech(samples) {
			t.Error("Expected silence to be NOT speech")
		}
	})

	t.Run("Speech-like Sample", func(t *testing.T) {
		vad := NewVAD(0.1, 0.5, 300)
		// High energy, low ZCR
		samples := make([]float32, 1600)
		for i := range samples {
			if (i/20)%2 == 0 {
				samples[i] = 0.5
			} else {
				samples[i] = -0.5
			}
		}
		if !vad.IsSpeech(samples) {
			t.Error("Expected speech-like sample to be speech")
		}
	})

	t.Run("Noise-like Sample (High ZCR)", func(t *testing.T) {
		vad := NewVAD(0.1, 0.5, 300)
		// High energy, high ZCR
		samples := make([]float32, 1600)
		for i := range samples {
			if i%2 == 0 {
				samples[i] = 0.5
			} else {
				samples[i] = -0.5
			}
		}
		rms := CalculateRMS(samples)
		zcr := CalculateZCR(samples)
		// ZCR will be 1599 / 1599 = 1.0, which is >= threshold 0.5
		if vad.IsSpeech(samples) {
			t.Errorf("Expected high-frequency noise to be NOT speech (rms=%f, zcr=%f)", rms, zcr)
		}
	})
}
