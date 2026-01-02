package features

import (
	"github.com/vitalii/hotword/pkg/audio"
	"github.com/vitalii/hotword/pkg/model"
)

// Extract converts raw audio samples into a flattened Mel-Spectrogram feature tensor.
func Extract(samples []float32, sampleRate, windowSize, hopSize, numMelFilters int) *model.Tensor {
	// 1. STFT
	stft := audio.STFT(samples, windowSize, hopSize)
	if len(stft) == 0 {
		return nil
	}

	// 2. Mel-Spectrogram
	melSpec := audio.MelSpectrogram(stft, numMelFilters, windowSize, sampleRate, 0, float64(sampleRate/2))

	// 3. Flatten into Tensor
	numFrames := len(melSpec)
	inputSize := numFrames * numMelFilters
	tensor := model.NewTensor([]int{inputSize})

	for i := 0; i < numFrames; i++ {
		for j := 0; j < numMelFilters; j++ {
			tensor.Data[i*numMelFilters+j] = melSpec[i][j]
		}
	}

	return tensor
}
