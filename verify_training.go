package main

import (
	"fmt"
	"github.com/vitalii/hotword/pkg/audio"
	"github.com/vitalii/hotword/pkg/model"
	"github.com/vitalii/hotword/pkg/train"
)

func main() {
	fmt.Println("Starting Training Pipeline Integration Verification...")

	// 1. Prepare dummy dataset
	// In a real scenario, use train.LoadDataset("data/hotword", "data/background")
	ds := &train.Dataset{
		Samples: []train.Sample{
			{Audio: make([]float32, 16000), IsHotword: true},
			{Audio: make([]float32, 16000), IsHotword: false},
		},
	}
	// Add some data so features aren't zero
	for i := range ds.Samples[0].Audio {
		ds.Samples[0].Audio[i] = 0.5
	}

	// 2. Define feature extractor (Mel-Spectrogram -> Flattened)
	sampleRate := 16000
	windowSize := 512
	hopSize := 256
	numMelFilters := 40
	
extractor := func(samples []float32) *model.Tensor {
		stft := audio.STFT(samples, windowSize, hopSize)
		melSpec := audio.MelSpectrogram(stft, numMelFilters, windowSize, sampleRate, 0, 8000)
		
		numFrames := len(melSpec)
		inputSize := numFrames * numMelFilters
		input := model.NewTensor([]int{inputSize})
		
		for i := 0; i < numFrames; i++ {
			for j := 0; j < numMelFilters; j++ {
				input.Data[i*numMelFilters+j] = melSpec[i][j]
			}
		}
		return input
	}

	// 3. Initialize Model and Trainer
	// Calculate input size from one pass
	dummyInput := extractor(ds.Samples[0].Audio)
	inputSize := len(dummyInput.Data)
	fmt.Printf("Model Input Size: %d\n", inputSize)

	weights := model.NewTensor([]int{1, inputSize})
	// Initialize weights with small random-ish values
	for i := range weights.Data {
		weights.Data[i] = 0.01
	}
	bias := []float32{0.0}
	
trainer := train.NewTrainer(weights, bias, 0.01)

	// 4. Run Training
	fmt.Println("Running training for 5 epochs...")
	trainer.Train(ds, 5, extractor)

	fmt.Println("\nVerification complete. Training loop ran successfully.")
}
