package model

import (
	"testing"
)

func TestConv2DBackward(t *testing.T) {
	// Simple case: 1 filter, 1 input channel, 3x3 input, 2x2 kernel, stride 1, padding 0
	input := NewTensor([]int{1, 3, 3})
	for i := range input.Data {
		input.Data[i] = float32(i + 1) // 1, 2, 3, 4, 5, 6, 7, 8, 9
	}
	
	weights := NewTensor([]int{1, 1, 2, 2})
	for i := range weights.Data {
		weights.Data[i] = 1.0
	}
	bias := []float32{0.0}
	
	// gradOutput: [1, 2, 2]
	gradOutput := NewTensor([]int{1, 2, 2})
	for i := range gradOutput.Data {
		gradOutput.Data[i] = 1.0
	}
	
	gradInput, gradWeights, gradBias := Conv2DBackward(input, weights, bias, gradOutput, 1, 0)
	
	// Check gradInput shape
	if gradInput.Shape[1] != 3 || gradInput.Shape[2] != 3 {
		t.Errorf("gradInput: expected 3x3, got %dx%d", gradInput.Shape[1], gradInput.Shape[2])
	}

	// gradBias = sum(gradOutput) = 1+1+1+1 = 4
	if gradBias[0] != 4.0 {
		t.Errorf("gradBias: expected 4.0, got %f", gradBias[0])
	}
	
	// gradWeights = conv(input, gradOutput)
	// weight[0,0,0,0] = in[0,0]*go[0,0] + in[0,1]*go[0,1] + in[1,0]*go[1,0] + in[1,1]*go[1,1]
	// weight[0,0,0,0] = 1*1 + 2*1 + 4*1 + 5*1 = 12
	if gradWeights.Data[0] != 12.0 {
		t.Errorf("gradWeights[0]: expected 12.0, got %f", gradWeights.Data[0])
	}
}
