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
