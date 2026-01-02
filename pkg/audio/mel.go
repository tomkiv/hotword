package audio

import (
	"math"
)

// HzToMel converts a frequency in Hz to the Mel scale.
func HzToMel(hz float64) float64 {
	return 2595 * math.Log10(1+hz/700)
}

// MelToHz converts a value on the Mel scale back to Hz.
func MelToHz(mel float64) float64 {
	return 700 * (math.Pow(10, mel/2595) - 1)
}

// CreateMelFilterbank creates a set of triangular filters equally spaced on the Mel scale.
func CreateMelFilterbank(numFilters, fftSize, sampleRate int, minHz, maxHz float64) [][]float32 {
	numBins := fftSize/2 + 1
	minMel := HzToMel(minHz)
	maxMel := HzToMel(maxHz)

	melPoints := make([]float64, numFilters+2)
	for i := 0; i < numFilters+2; i++ {
		melPoints[i] = minMel + float64(i)*(maxMel-minMel)/float64(numFilters+1)
	}

	hzPoints := make([]float64, numFilters+2)
	for i := 0; i < numFilters+2; i++ {
		hzPoints[i] = MelToHz(melPoints[i])
	}

	binPoints := make([]int, numFilters+2)
	for i := 0; i < numFilters+2; i++ {
		binPoints[i] = int(math.Floor(float64(fftSize+1) * hzPoints[i] / float64(sampleRate)))
	}

	filters := make([][]float32, numFilters)
	for i := 0; i < numFilters; i++ {
		filter := make([]float32, numBins)
		startBin := binPoints[i]
		midBin := binPoints[i+1]
		endBin := binPoints[i+2]

		for j := startBin; j < midBin; j++ {
			if j >= 0 && j < numBins {
				if midBin != startBin {
					filter[j] = float32(j-startBin) / float32(midBin-startBin)
				}
			}
		}
		for j := midBin; j < endBin; j++ {
			if j >= 0 && j < numBins {
				if endBin != midBin {
					filter[j] = float32(endBin-j) / float32(endBin-midBin)
				}
			}
		}
		filters[i] = filter
	}

	return filters
}

// ApplyFilterbank applies the Mel-filterbank to a single STFT magnitude frame.
func ApplyFilterbank(stftFrame []float32, filterbank [][]float32) []float32 {
	melFrame := make([]float32, len(filterbank))
	for i, filter := range filterbank {
		var sum float32
		for j, weight := range filter {
			if j < len(stftFrame) {
				sum += stftFrame[j] * weight
			}
		}
		melFrame[i] = sum
	}
	return melFrame
}

// MelSpectrogram converts a full STFT spectrogram into a Mel-spectrogram.
func MelSpectrogram(spectrogram [][]float32, numFilters, fftSize, sampleRate int, minHz, maxHz float64) [][]float32 {
	fb := CreateMelFilterbank(numFilters, fftSize, sampleRate, minHz, maxHz)
	melSpectrogram := make([][]float32, len(spectrogram))
	for i, frame := range spectrogram {
		melSpectrogram[i] = ApplyFilterbank(frame, fb)
	}
	return melSpectrogram
}