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

func TestShift(t *testing.T) {
	samples := []float32{1, 2, 3, 4, 5}
	
	t.Run("Forward Shift", func(t *testing.T) {
		shifted := Shift(samples, 2)
		expected := []float32{4, 5, 1, 2, 3}
		for i := range shifted {
			if shifted[i] != expected[i] {
				t.Errorf("At index %d: expected %f, got %f", i, expected[i], shifted[i])
			}
		}
	})

	t.Run("Backward Shift", func(t *testing.T) {
		shifted := Shift(samples, -1)
		expected := []float32{2, 3, 4, 5, 1}
		for i := range shifted {
			if shifted[i] != expected[i] {
				t.Errorf("At index %d: expected %f, got %f", i, expected[i], shifted[i])
			}
		}
	})
}

func TestScale(t *testing.T) {
	samples := []float32{1.0, -1.0, 0.5}
	scaled := Scale(samples, 0.5)
	expected := []float32{0.5, -0.5, 0.25}
	
	for i := range scaled {
		if math.Abs(float64(scaled[i]-expected[i])) > 1e-6 {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], scaled[i])
		}
	}
}
