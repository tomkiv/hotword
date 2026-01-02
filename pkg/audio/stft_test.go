package audio

import (
	"math"
	"testing"
)

func TestSTFT(t *testing.T) {
	t.Run("Hamming Window", func(t *testing.T) {
		window := HammingWindow(1025)
		if len(window) != 1025 {
			t.Errorf("Expected window size 1025, got %d", len(window))
		}
		// Hamming window at center should be 1.0
		if math.Abs(float64(window[512])-1.0) > 1e-6 {
			t.Errorf("Expected window[512] to be ~1.0, got %f", window[512])
		}
		// Hamming window at edges should be 0.08
		if math.Abs(float64(window[0])-0.08) > 1e-6 {
			t.Errorf("Expected window[0] to be ~0.08, got %f", window[0])
		}
	})

	t.Run("STFT Shape", func(t *testing.T) {
		samples := make([]float32, 16000) // 1 second at 16kHz
		for i := range samples {
			samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / 16000))
		}

		windowSize := 512
		hopSize := 256
		spectrogram := STFT(samples, windowSize, hopSize)

		// Expected number of frames: (16000 - 512) / 256 + 1 = 61.5 -> 61 frames
		// Actually, let's see how many frames we expect:
		// floor((len - win) / hop) + 1
		expectedFrames := (len(samples)-windowSize)/hopSize + 1
		if len(spectrogram) != expectedFrames {
			t.Errorf("Expected %d frames, got %d", expectedFrames, len(spectrogram))
		}

		if len(spectrogram[0]) != windowSize/2+1 {
			t.Errorf("Expected %d bins, got %d", windowSize/2+1, len(spectrogram[0]))
		}
	})
}
