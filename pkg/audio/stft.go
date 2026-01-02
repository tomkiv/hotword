package audio

import (
	"math"
	"math/cmplx"
)

// HammingWindow returns a Hamming window of the specified size.
func HammingWindow(size int) []float32 {
	window := make([]float32, size)
	for i := 0; i < size; i++ {
		window[i] = float32(0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(size-1)))
	}
	return window
}

// STFT performs Short-Time Fourier Transform on the input samples.
// It returns a 2D slice where each inner slice is the magnitude spectrum of a frame.
func STFT(samples []float32, windowSize, hopSize int) [][]float32 {
	numFrames := (len(samples) - windowSize) / hopSize + 1
	if numFrames <= 0 {
		return nil
	}

	window := HammingWindow(windowSize)
	spectrogram := make([][]float32, numFrames)

	for i := 0; i < numFrames; i++ {
		start := i * hopSize
		frame := make([]complex128, windowSize)
		for j := 0; j < windowSize; j++ {
			frame[j] = complex(float64(samples[start+j]*window[j]), 0)
		}

		fftResult := fft(frame)
		
		// Take magnitude of the first half (positive frequencies)
		numBins := windowSize/2 + 1
		magnitudeFrame := make([]float32, numBins)
		for j := 0; j < numBins; j++ {
			magnitudeFrame[j] = float32(cmplx.Abs(fftResult[j]))
		}
		spectrogram[i] = magnitudeFrame
	}

	return spectrogram
}

// fft performs a basic Cooley-Tukey FFT.
// Input size must be a power of 2.
func fft(a []complex128) []complex128 {
	n := len(a)
	if n <= 1 {
		return a
	}

	// Pad to power of 2 if necessary (though STFT windowSize is usually power of 2)
	// For this implementation, we assume n is a power of 2.

	even := make([]complex128, n/2)
	odd := make([]complex128, n/2)
	for i := 0; i < n/2; i++ {
		even[i] = a[2*i]
		odd[i] = a[2*i+1]
	}

	evenFFT := fft(even)
	oddFFT := fft(odd)

	result := make([]complex128, n)
	for k := 0; k < n/2; k++ {
		t := cmplx.Exp(complex(0, -2*math.Pi*float64(k)/float64(n))) * oddFFT[k]
		result[k] = evenFFT[k] + t
		result[k+n/2] = evenFFT[k] - t
	}
	return result
}
