package capture

import (
	"strings"
	"testing"
)

func TestCalculateLevels(t *testing.T) {
	samples := []float32{0.5, -0.5, 0.5, -0.5}
	rms, peak := CalculateLevels(samples)
	
	if rms != 0.5 {
		t.Errorf("Expected RMS 0.5, got %f", rms)
	}
	if peak != 0.5 {
		t.Errorf("Expected peak 0.5, got %f", peak)
	}
}

func TestGenerateVUBar(t *testing.T) {
	bar := GenerateVUBar(0.5, 20)
	expected := "[##########          ]"
	if bar != expected {
		t.Errorf("Expected bar %s, got %s", expected, bar)
	}
	
	if !strings.HasPrefix(bar, "[") || !strings.HasSuffix(bar, "]") {
		t.Error("Bar missing brackets")
	}
}
