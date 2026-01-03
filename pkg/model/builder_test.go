package model

import (
	"testing"
)

func TestBuildModelFromConfig(t *testing.T) {
	configs := []LayerConfig{
		{Type: "conv2d", Filters: 8, KernelSize: 3, Stride: 1, Padding: 1},
		{Type: "relu"},
		{Type: "maxpool2d", KernelSize: 2, Stride: 2},
		{Type: "dense", Units: 1},
		{Type: "sigmoid"},
	}
	
	// Input shape: [1, 61, 40] (1 channel, 61 frames, 40 mel filters)
	inputShape := []int{1, 61, 40}
	
	m, err := BuildModelFromConfig(configs, inputShape)
	if err != nil {
		t.Fatalf("Failed to build model: %v", err)
	}
	
	if len(m.Layers) != 5 {
		t.Errorf("Expected 5 layers, got %d", len(m.Layers))
	}
	
	// Check first layer params (Xavier init)
	w, b := m.Layers[0].Params()
	if w == nil || b == nil {
		t.Error("Expected weights and bias for first layer")
	}
	
	// Verify output of forward pass
	input := NewTensor(inputShape)
	for i := range input.Data {
		input.Data[i] = 0.1
	}
	
	output := m.Forward(input)
	if output == nil {
		t.Fatal("Forward pass returned nil")
	}
	
	if len(output.Data) != 1 {
		t.Errorf("Expected output size 1, got %d", len(output.Data))
	}
	
	if output.Data[0] < 0 || output.Data[0] > 1 {
		t.Errorf("Expected probability in [0, 1], got %f", output.Data[0])
	}
}
