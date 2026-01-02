package audio

import (
	"math"
	"testing"
)

func TestMixNoise(t *testing.T) {
	t.Run("Mix 50/50", func(t *testing.T) {
		signal := []float32{1.0, 1.0, 1.0}
		noise := []float32{0.0, 0.0, 0.0}
		
		mixed := MixNoise(signal, noise, 0.5)
		
		// expected = signal*0.5 + noise*0.5 = 0.5
		for i := range mixed {
			if math.Abs(float64(mixed[i]-0.5)) > 1e-6 {
				t.Errorf("At index %d: expected 0.5, got %f", i, mixed[i])
			}
		}
	})

	t.Run("Different Lengths", func(t *testing.T) {
		signal := []float32{1.0, 1.0}
		noise := []float32{0.0, 0.0, 0.0, 0.0}
		
		mixed := MixNoise(signal, noise, 0.1) // ratio is noise ratio
		
		if len(mixed) != len(signal) {
			t.Errorf("Expected length %d, got %d", len(signal), len(mixed))
		}
		
		// expected = signal*(1-0.1) + noise*0.1 = 0.9 + 0.0 = 0.9
		for i := range mixed {
			if math.Abs(float64(mixed[i]-0.9)) > 1e-6 {
				t.Errorf("At index %d: expected 0.9, got %f", i, mixed[i])
			}
		}
	})
}
