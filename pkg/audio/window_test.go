package audio

import (
	"testing"
)

func TestSlidingWindow(t *testing.T) {
	windowSize := 10
	hopSize := 5
	sw := NewSlidingWindow(windowSize, hopSize)

	t.Run("Insufficient Data", func(t *testing.T) {
		samples := make([]float32, 8)
		sw.AddSamples(samples)
		window, ok := sw.NextWindow()
		if ok || window != nil {
			t.Error("Expected no window for insufficient data")
		}
	})

	t.Run("Produce Windows", func(t *testing.T) {
		// New sw to reset state
		sw = NewSlidingWindow(windowSize, hopSize)
		
		// Add 15 samples: Should produce 2 windows (0-10, 5-15)
		samples := make([]float32, 15)
		for i := range samples {
			samples[i] = float32(i)
		}
		sw.AddSamples(samples)

		w1, ok1 := sw.NextWindow()
		if !ok1 || len(w1) != 10 {
			t.Fatalf("Expected first window of size 10")
		}
		if w1[0] != 0 || w1[9] != 9 {
			t.Errorf("First window data mismatch")
		}

		w2, ok2 := sw.NextWindow()
		if !ok2 || len(w2) != 10 {
			t.Fatalf("Expected second window of size 10")
		}
		if w2[0] != 5 || w2[9] != 14 {
			t.Errorf("Second window data mismatch")
		}

		_, ok3 := sw.NextWindow()
		if ok3 {
			t.Error("Expected no more windows")
		}
	})
}
