package model

import (
	"testing"
)

func TestReLUBackward(t *testing.T) {
	// Input: [2]
	input := &Tensor{Data: []float32{-1.0, 2.0}, Shape: []int{2}}
	// gradOutput: [2]
	gradOutput := &Tensor{Data: []float32{1.0, 1.0}, Shape: []int{2}}
	
	gradInput := ReLUBackward(input, gradOutput)
	
	// gradInput[0] = 0.0 (because input[0] <= 0)
	if gradInput.Data[0] != 0.0 {
		t.Errorf("gradInput[0]: expected 0.0, got %f", gradInput.Data[0])
	}
	// gradInput[1] = 1.0 (because input[1] > 0)
	if gradInput.Data[1] != 1.0 {
		t.Errorf("gradInput[1]: expected 1.0, got %f", gradInput.Data[1])
	}
}

func TestMaxPool2DBackward(t *testing.T) {
	// Input: [1, 2, 2]
	input := NewTensor([]int{1, 2, 2})
	input.Set([]int{0, 0, 0}, 1.0)
	input.Set([]int{0, 0, 1}, 5.0)
	input.Set([]int{0, 1, 0}, 3.0)
	input.Set([]int{0, 1, 1}, 2.0)
	
	// gradOutput: [1, 1, 1] (pooling 2x2 to 1x1)
	gradOutput := &Tensor{Data: []float32{10.0}, Shape: []int{1, 1, 1}}
	
	gradInput := MaxPool2DBackward(input, gradOutput, 2, 2)
	
	// gradInput[0, 0, 1] should be 10.0 (the max element), others 0
	if gradInput.Get([]int{0, 0, 1}) != 10.0 {
		t.Errorf("gradInput[0,0,1]: expected 10.0, got %f", gradInput.Get([]int{0, 0, 1}))
	}
	if gradInput.Get([]int{0, 0, 0}) != 0.0 {
		t.Errorf("gradInput[0,0,0]: expected 0.0, got %f", gradInput.Get([]int{0, 0, 0}))
	}
}
