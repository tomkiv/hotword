package main

import (
	"fmt"
	"os"
	"github.com/tomkiv/hotword/pkg/model"
)

func main() {
	fmt.Println("Starting Persistence Verification...")

	// 1. Create a dummy model
	weights := model.NewTensor([]int{1, 5})
	for i := range weights.Data {
		weights.Data[i] = float32(i) * 0.1
	}
	bias := []float32{0.5}

	m := model.NewSequentialModel(
		model.NewDenseLayer(weights, bias),
		model.NewSigmoidLayer(),
	)

	fmt.Printf("Original Weights: %v\n", weights.Data)
	fmt.Printf("Original Bias:    %v\n", bias)

	// 2. Save to file
	modelPath := "/Users/vitalii/.gemini/tmp/4ea7f44ea8ed39b191b36db796e7cfa745b89b5eb30827822379ef6ae57a83d0/verify_model.bin"
	if err := model.SaveModel(modelPath, m); err != nil {
		fmt.Printf("Error saving model: %v\n", err)
		return
	}
	defer os.Remove(modelPath)

	// 3. Load back
	loadedModel, err := model.LoadModel(modelPath)
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		return
	}

	// 4. Comparison
	layers := loadedModel.GetLayers()
	if len(layers) < 1 {
		fmt.Println("Error: Loaded model has no layers")
		return
	}
	
lWeights, lBias := layers[0].Params()

	fmt.Printf("Loaded Weights:   %v\n", lWeights.Data)
	fmt.Printf("Loaded Bias:      %v\n", lBias)

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