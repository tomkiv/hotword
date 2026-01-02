package model

import (
	"math"
	"testing"
)

func TestDenseBackward(t *testing.T) {
	// Input: [2]
	input := &Tensor{Data: []float32{1.0, 2.0}, Shape: []int{2}}
	// Weights: [1, 2]
	weights := &Tensor{Data: []float32{0.5, -0.5}, Shape: []int{1, 2}}
	// Bias: [1]
	bias := []float32{0.1}
	
	// gradOutput: [1] (dL/dY)
	gradOutput := &Tensor{Data: []float32{1.0}, Shape: []int{1}}
	
	gradInput, gradWeights, gradBias := DenseBackward(input, weights, bias, gradOutput)
	
	// dL/dW = gradOutput * input^T
	// gradWeights[0] = 1.0 * 1.0 = 1.0
	// gradWeights[1] = 1.0 * 2.0 = 2.0
	if math.Abs(float64(gradWeights.Data[0]-1.0)) > 1e-6 {
		t.Errorf("gradWeights[0]: expected 1.0, got %f", gradWeights.Data[0])
	}
	if math.Abs(float64(gradWeights.Data[1]-2.0)) > 1e-6 {
		t.Errorf("gradWeights[1]: expected 2.0, got %f", gradWeights.Data[1])
	}
	
	// dL/dB = gradOutput
	if math.Abs(float64(gradBias[0]-1.0)) > 1e-6 {
		t.Errorf("gradBias[0]: expected 1.0, got %f", gradBias[0])
	}
	
	// dL/dX = W^T * gradOutput
	// gradInput[0] = 0.5 * 1.0 = 0.5
	// gradInput[1] = -0.5 * 1.0 = -0.5
	if math.Abs(float64(gradInput.Data[0]-0.5)) > 1e-6 {
		t.Errorf("gradInput[0]: expected 0.5, got %f", gradInput.Data[0])
	}
	if math.Abs(float64(gradInput.Data[1]-(-0.5))) > 1e-6 {
		t.Errorf("gradInput[1]: expected -0.5, got %f", gradInput.Data[1])
	}
}
