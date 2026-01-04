package main

import (
	"fmt"

	"github.com/tomkiv/hotword/pkg/audio"
	"github.com/tomkiv/hotword/pkg/engine"
	"github.com/tomkiv/hotword/pkg/model"
)

func main() {
	fmt.Println("Starting End-to-End Integration Verification...")

	sampleRate := 16000
	windowSize := 512
	hopSize := 256
	numMelFilters := 40

	// 1. Create a dummy model
	// We'll simulate 1 second worth of audio (approx 61 frames)
	numSamples := 16000
	numFrames := (numSamples-windowSize)/hopSize + 1
	inputSize := numFrames * numMelFilters

	weights := model.NewTensor([]int{1, inputSize})
	for i := range weights.Data {
		weights.Data[i] = 0.001
	}
	bias := []float32{0.1}

	// 2. Initialize Engine
	m := model.NewSequentialModel(
		model.NewDenseLayer(weights, bias),
		model.NewSigmoidLayer(),
	)
	e := engine.NewEngine(m, sampleRate)

	// 3. Initialize Sliding Window
	sw := audio.NewSlidingWindow(numSamples, hopSize)

	// 4. Simulate incoming audio stream (e.g., 2 seconds of audio)
	totalSamples := make([]float32, sampleRate*2)
	for i := range totalSamples {
		totalSamples[i] = 0.1 // Just some background noise
	}

	fmt.Println("Feeding audio into sliding window...")
	sw.AddSamples(totalSamples)

	// 5. Process windows from the stream
	count := 0
	for {
		chunk, ok := sw.NextWindow()
		if !ok {
			break
		}

		prob, detected := e.Process(chunk, 0.5)
		count++
		fmt.Printf("Processed Chunk %d: Probability = %.4f, Detected = %v\n", count, prob, detected)
	}

	if count > 0 {
		fmt.Printf("Total Chunks Processed: %d\n", count)
		fmt.Println("End-to-End Integration Verification Successful!")
	} else {
		fmt.Println("Verification Failed: No chunks were processed.")
	}
}
