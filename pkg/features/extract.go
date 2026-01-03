package features

import (
	"math"

	"github.com/vitalii/hotword/pkg/audio"
	"github.com/vitalii/hotword/pkg/model"
)

// Extract converts raw audio samples into a flattened Mel-Spectrogram feature tensor.
func Extract(samples []float32, sampleRate, windowSize, hopSize, numMelFilters int) *model.Tensor {
	// 0. Pre-emphasis
	// Apply pre-emphasis to filter out low-frequency noise (DC offset, hum)
	// and flatten the spectral tilt.
	samples = audio.PreEmphasis(samples, 0.97)

	// 1. STFT
	stft := audio.STFT(samples, windowSize, hopSize)
	if len(stft) == 0 {
		return nil
	}

	// 2. Mel-Spectrogram
	melSpec := audio.MelSpectrogram(stft, numMelFilters, windowSize, sampleRate, 0, float64(sampleRate/2))

	// 3. Reshape into 3D Tensor [1, numFrames, numMelFilters] and apply Log-scaling
	numFrames := len(melSpec)
	tensor := model.NewTensor([]int{1, numFrames, numMelFilters})

	for i := 0; i < numFrames; i++ {
		for j := 0; j < numMelFilters; j++ {
			// Apply Log-scaling: log(1 + 1000*x)
			val := melSpec[i][j]
			tensor.Set([]int{0, i, j}, float32(math.Log1p(float64(val)*1000.0)))
		}
	}

	return tensor
}
