package model

import (
	"math"
	"testing"
)

func TestSGDOptimizer(t *testing.T) {
	// Simple weights: [2]
	weights := &Tensor{Data: []float32{1.0, 2.0}, Shape: []int{2}}
	// Simple gradient: [2]
	gradWeights := &Tensor{Data: []float32{0.1, -0.5}, Shape: []int{2}}
	
	learningRate := float32(0.1)
	
	// SGD Update: W = W - LR * gradW
	SGDUpdate(weights, gradWeights, learningRate)
	
	// W[0] = 1.0 - 0.1 * 0.1 = 1.0 - 0.01 = 0.99
	// W[1] = 2.0 - 0.1 * (-0.5) = 2.0 + 0.05 = 2.05
	
	if math.Abs(float64(weights.Data[0]-0.99)) > 1e-6 {
		t.Errorf("W[0]: expected 0.99, got %f", weights.Data[0])
	}
	if math.Abs(float64(weights.Data[1]-2.05)) > 1e-6 {
		t.Errorf("W[1]: expected 2.05, got %f", weights.Data[1])
	}
}

func TestSGDBiasUpdate(t *testing.T) {
	bias := []float32{0.5, -0.1}
	gradBias := []float32{0.1, 0.2}
	lr := float32(0.1)

	SGDBiasUpdate(bias, gradBias, lr)

	// B[0] = 0.5 - 0.1 * 0.1 = 0.49
	// B[1] = -0.1 - 0.1 * 0.2 = -0.12
	if math.Abs(float64(bias[0]-0.49)) > 1e-6 {
		t.Errorf("B[0]: expected 0.49, got %f", bias[0])
	}
	if math.Abs(float64(bias[1]-(-0.12))) > 1e-6 {
		t.Errorf("B[1]: expected -0.12, got %f", bias[1])
	}
}
