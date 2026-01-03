package model

import (
	"testing"
)

func TestSequentialModelForward(t *testing.T) {
	// Create a simple model: Dense(2->2) -> ReLU -> Dense(2->1) -> Sigmoid
	w1 := NewTensor([]int{2, 2})
	w1.Data = []float32{1, 0, 0, 1} // Identity
	b1 := []float32{0, 0}
	
	w2 := NewTensor([]int{1, 2})
	w2.Data = []float32{1, 1}
	b2 := []float32{0}
	
	m := NewSequentialModel(
		NewDenseLayer(w1, b1),
		NewReLULayer(),
		NewDenseLayer(w2, b2),
		NewSigmoidLayer(),
	)
	
	input := NewTensor([]int{2})
	input.Data = []float32{0.5, -0.5}
	
	// Step by step:
	// 1. Dense: [0.5, -0.5] * I + 0 = [0.5, -0.5]
	// 2. ReLU: [0.5, 0]
	// 3. Dense: [0.5, 0] * [1, 1]^T + 0 = 0.5
	// 4. Sigmoid: 1 / (1 + exp(-0.5)) ~= 0.622459
	
	output := m.Forward(input)
	
	if len(output.Data) != 1 {
		t.Fatalf("Expected output size 1, got %d", len(output.Data))
	}
	
	expected := float32(0.622459)
	if mathAbs(output.Data[0]-expected) > 1e-5 {
		t.Errorf("Expected %f, got %f", expected, output.Data[0])
	}
}

func mathAbs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
