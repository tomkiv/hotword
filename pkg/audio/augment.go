package audio

// MixNoise mixes a noise buffer into a signal buffer.
// noiseRatio determines the amount of noise (0.0 to 1.0).
// If buffers have different lengths, it uses the shortest length.
func MixNoise(signal, noise []float32, noiseRatio float32) []float32 {
	length := len(signal)
	if len(noise) < length {
		length = len(noise)
	}

	mixed := make([]float32, length)
	signalRatio := 1.0 - noiseRatio

	for i := 0; i < length; i++ {
		mixed[i] = signal[i]*signalRatio + noise[i]*noiseRatio
	}

	return mixed
}

// Shift performs a circular shift on the provided samples.
func Shift(samples []float32, offset int) []float32 {
	n := len(samples)
	if n == 0 {
		return samples
	}

	// Normalize offset to [0, n)
	offset = offset % n
	if offset < 0 {
		offset += n
	}

	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[(i+offset)%n] = samples[i]
	}
	return out
}

// Scale multiplies the amplitude of the signal by the provided gain.
func Scale(samples []float32, gain float32) []float32 {
	out := make([]float32, len(samples))
	for i, s := range samples {
		out[i] = s * gain
	}
	return out
}