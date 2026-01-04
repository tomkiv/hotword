package main

import (
	"fmt"
	"github.com/tomkiv/hotword/pkg/model"
)

func main() {
	fmt.Println("Starting Model Verification...")

	// 1. Create dummy input: 1 channel, 10x10 spectrogram
	input := model.NewTensor([]int{1, 10, 10})
	for i := range input.Data {
		input.Data[i] = 0.5
	}
	fmt.Printf("Input Shape: %v\n", input.Shape)

	// 2. Conv Layer: 2 filters, 3x3 kernel, stride 1, padding 1
	convWeights := model.NewTensor([]int{2, 1, 3, 3})
	for i := range convWeights.Data {
		convWeights.Data[i] = 0.1
	}
	convBias := []float32{0.0, 0.0}
	
	convOut := model.Conv2D(input, convWeights, convBias, 1, 1)
	fmt.Printf("After Conv Shape: %v\n", convOut.Shape)

	// 3. ReLU
	reluOut := model.ReLU(convOut)

	// 4. MaxPool: 2x2, stride 2
	poolOut := model.MaxPool2D(reluOut, 2, 2)
	fmt.Printf("After Pool Shape: %v\n", poolOut.Shape)

	// 5. Dense: Output 1 unit (probability)
	// Input size to dense = channels * h * w = 2 * 5 * 5 = 50
	denseWeights := model.NewTensor([]int{1, 50})
	for i := range denseWeights.Data {
		denseWeights.Data[i] = 0.01
	}
	denseBias := []float32{0.1}

	finalOut := model.Dense(poolOut, denseWeights, denseBias)
	fmt.Printf("Final Output (Probability): %.4f\n", finalOut.Data[0])

	if len(finalOut.Data) == 1 {
		fmt.Println("Verification Successful!")
	}
}
