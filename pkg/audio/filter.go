package audio

// PreEmphasis applies a pre-emphasis filter to the audio samples.
// y[n] = x[n] - coeff * x[n-1]
// This amplifies high frequencies and helps to balance the frequency spectrum,
// making the model more robust to low-frequency noise.
// A common coefficient value is 0.97.
func PreEmphasis(samples []float32, coeff float32) []float32 {
	if len(samples) == 0 {
		return nil
	}

	out := make([]float32, len(samples))
	out[0] = samples[0]

	for i := 1; i < len(samples); i++ {
		out[i] = samples[i] - coeff*samples[i-1]
	}

	return out
}
