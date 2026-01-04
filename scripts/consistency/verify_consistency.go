package main

import (
	"fmt"
	"os"
	"github.com/tomkiv/hotword/pkg/model"
	"github.com/tomkiv/hotword/pkg/train"
)

func main() {
	fmt.Println("Starting Consistency Verification...")

	// 1. Setup a tiny model
	weights := model.NewTensor([]int{1, 5})
	for i := range weights.Data {
		weights.Data[i] = 0.1
	}
	bias := []float32{0.0}
	
m := model.NewSequentialModel(
		model.NewDenseLayer(weights, bias),
		model.NewSigmoidLayer(),
	)

	// 2. Mock a single training step to get non-initial weights
	ds := &train.Dataset{
		Samples: []train.Sample{
			{Audio: []float32{1, 1, 1, 1, 1}, IsHotword: true},
		},
	}
	extractor := func(a []float32) *model.Tensor {
		return &model.Tensor{Data: a, Shape: []int{5}}
	}
	trainer := train.NewTrainer(m, 0.1)
	trainer.Train(ds, 1, extractor)

	// 3. Inference with original
	input := &model.Tensor{Data: []float32{0.5, 0.5, 0.5, 0.5, 0.5}, Shape: []int{5}}
	origOut := m.Forward(input)
	origVal := origOut.Data[0]

	// 4. Save and Load
	path := "/Users/vitalii/.gemini/tmp/4ea7f44ea8ed39b191b36db796e7cfa745b89b5eb30827822379ef6ae57a83d0/consistency_model.bin"
	if err := model.SaveModel(path, m); err != nil {
		fmt.Printf("Error saving: %v\n", err)
		return
	}
	defer os.Remove(path)

	loadedModel, err := model.LoadModel(path)
	if err != nil {
		fmt.Printf("Error loading: %v\n", err)
		return
	}

	// 5. Inference with loaded
	loadOut := loadedModel.Forward(input)
	loadVal := loadOut.Data[0]

	fmt.Printf("Original Inference: %f\n", origVal)
	fmt.Printf("Reloaded Inference: %f\n", loadVal)

	if origVal == loadVal {
		fmt.Println("End-to-end consistency verified!")
	} else {
		fmt.Println("End-to-end consistency FAILED!")
	}
}