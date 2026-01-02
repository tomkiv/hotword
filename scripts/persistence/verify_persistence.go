package main

import (
	"fmt"
	"os"
	"github.com/vitalii/hotword/pkg/model"
)

func main() {
	fmt.Println("Starting Phase 1 Persistence Verification...")

	// 1. Create a dummy model
	weights := model.NewTensor([]int{1, 5})
	for i := range weights.Data {
		weights.Data[i] = float32(i) * 0.1
	}
	bias := []float32{0.5}

	fmt.Printf("Original Weights: %v\n", weights.Data)
	fmt.Printf("Original Bias:    %v\n", bias)

	// 2. Save to file
	modelPath := "/Users/vitalii/.gemini/tmp/4ea7f44ea8ed39b191b36db796e7cfa745b89b5eb30827822379ef6ae57a83d0/verify_model.bin"
	if err := model.SaveModel(modelPath, weights, bias); err != nil {
		fmt.Printf("Error saving model: %v\n", err)
		return
	}
	defer os.Remove(modelPath)

	// 3. Load back
	lWeights, lBias, err := model.LoadModel(modelPath)
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		return
	}

	fmt.Printf("Loaded Weights:   %v\n", lWeights.Data)
	fmt.Printf("Loaded Bias:      %v\n", lBias)

	// 4. Comparison
	match := true
	for i := range weights.Data {
		if weights.Data[i] != lWeights.Data[i] {
			match = false
			break
		}
	}
	if bias[0] != lBias[0] {
		match = false
	}

	if match {
		fmt.Println("Persistence verification successful! Exact match found.")
	} else {
		fmt.Println("Persistence verification FAILED! Data mismatch.")
	}
}
