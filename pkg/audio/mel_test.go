package audio

import (
	"math"
	"testing"
)

func TestMelScale(t *testing.T) {
	t.Run("Hz to Mel and back", func(t *testing.T) {
		hz := 1000.0
		mel := HzToMel(hz)
		back := MelToHz(mel)
		if math.Abs(hz-back) > 1e-6 {
			t.Errorf("Expected %f, got %f", hz, back)
		}
	})
}

func TestMelFilterbank(t *testing.T) {
	numFilters := 40
	fftSize := 512
	sampleRate := 16000
	minHz := 0.0
	maxHz := 8000.0

	fb := CreateMelFilterbank(numFilters, fftSize, sampleRate, minHz, maxHz)

	if len(fb) != numFilters {
		t.Errorf("Expected %d filters, got %d", numFilters, len(fb))
	}

	for i, filter := range fb {
		if len(filter) != fftSize/2+1 {
			t.Errorf("Filter %d: Expected %d weights, got %d", i, fftSize/2+1, len(filter))
		}
	}
}

func TestMelSpectrogram(t *testing.T) {
	numFilters := 40
	fftSize := 512
	sampleRate := 16000
	
	fb := CreateMelFilterbank(numFilters, fftSize, sampleRate, 0, 8000)
	
	// Dummy STFT magnitude frame (512/2 + 1 = 257 bins)
	stftFrame := make([]float32, fftSize/2+1)
	for i := range stftFrame {
		stftFrame[i] = 1.0
	}
	
	melFrame := ApplyFilterbank(stftFrame, fb)
	if len(melFrame) != numFilters {
		t.Errorf("Expected %d mel bins, got %d", numFilters, len(melFrame))
	}
}
