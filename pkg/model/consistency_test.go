package model

import (
	"os"
	"testing"
)

func TestModelConsistency(t *testing.T) {
	t.Run("Save Load Inference Match", func(t *testing.T) {
		// Create a realistic-ish model: Conv2D -> ReLU -> Dense -> Sigmoid
		configs := []LayerConfig{
			{Type: "conv2d", Filters: 4, KernelSize: 3, Stride: 1, Padding: 1},
			{Type: "relu"},
			{Type: "dense", Units: 1},
			{Type: "sigmoid"},
		}
		inputShape := []int{1, 10, 10}
		
		mOrig, _ := BuildModelFromConfig(configs, inputShape)
		
		tmpFile := "consistency_test.bin"
		SaveModel(tmpFile, mOrig)
		defer os.Remove(tmpFile)
		
		mLoaded, err := LoadModel(tmpFile)
		if err != nil {
			t.Fatalf("Failed to load model: %v", err)
		}
		
		input := NewTensor(inputShape)
		for i := range input.Data {
			input.Data[i] = 0.123
		}
		
		outOrig := mOrig.Forward(input)
		outLoaded := mLoaded.Forward(input)
		
		if outOrig.Data[0] != outLoaded.Data[0] {
			t.Errorf("Consistency check failed: Original Prob=%f, Loaded Prob=%f", outOrig.Data[0], outLoaded.Data[0])
		}
	})
}