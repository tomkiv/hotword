package model

import (
	"testing"
)

func TestGRUForward(t *testing.T) {
	ResetRand(42) // For reproducibility

	inputSize := 10
	hiddenSize := 8
	seqLen := 5

	gru := NewGRULayer(inputSize, hiddenSize)

	// Create input sequence [seqLen, inputSize]
	input := NewTensor([]int{seqLen, inputSize})
	for i := range input.Data {
		input.Data[i] = randFloat32()*2 - 1
	}

	// Forward pass
	output := gru.Forward(input)

	// Check output shape
	if output == nil {
		t.Fatal("GRU forward returned nil")
	}

	if len(output.Data) != hiddenSize {
		t.Errorf("Expected output size %d, got %d", hiddenSize, len(output.Data))
	}

	// Check that output is not all zeros
	allZero := true
	for _, v := range output.Data {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("GRU output is all zeros")
	}
}

func TestGRUForward3D(t *testing.T) {
	ResetRand(42)

	// Test with 3D input (from CNN output)
	// Shape: [channels, height, width] = [8, 30, 20]
	// This should be reshaped to [30, 8*20] = [30, 160]
	channels := 8
	height := 30
	width := 20

	inputSize := channels * width // 160
	hiddenSize := 32

	gru := NewGRULayer(inputSize, hiddenSize)

	input := NewTensor([]int{channels, height, width})
	for i := range input.Data {
		input.Data[i] = randFloat32()*2 - 1
	}

	output := gru.Forward(input)

	if output == nil {
		t.Fatal("GRU forward with 3D input returned nil")
	}

	if len(output.Data) != hiddenSize {
		t.Errorf("Expected output size %d, got %d", hiddenSize, len(output.Data))
	}
}

func TestGRUBackward(t *testing.T) {
	ResetRand(42)

	inputSize := 4
	hiddenSize := 3
	seqLen := 2

	gru := NewGRULayer(inputSize, hiddenSize)

	input := NewTensor([]int{seqLen, inputSize})
	for i := range input.Data {
		input.Data[i] = randFloat32()*2 - 1
	}

	// Forward pass
	output := gru.Forward(input)

	// Create gradient from output
	gradOutput := NewTensor([]int{hiddenSize})
	for i := range gradOutput.Data {
		gradOutput.Data[i] = 1.0 // Simple gradient
	}

	// Backward pass
	gradInput, gradWeights, gradBias := gru.Backward(input, gradOutput)

	// Check gradient shapes
	if gradInput == nil {
		t.Fatal("Backward returned nil gradInput")
	}

	if len(gradInput.Data) != seqLen*inputSize {
		t.Errorf("Expected input gradient size %d, got %d", seqLen*inputSize, len(gradInput.Data))
	}

	if gradWeights == nil {
		t.Fatal("Backward returned nil gradWeights")
	}

	expectedWeightSize := 3*hiddenSize*inputSize + 3*hiddenSize*hiddenSize
	if len(gradWeights.Data) != expectedWeightSize {
		t.Errorf("Expected weight gradient size %d, got %d", expectedWeightSize, len(gradWeights.Data))
	}

	expectedBiasSize := hiddenSize * 3
	if len(gradBias) != expectedBiasSize {
		t.Errorf("Expected bias gradient size %d, got %d", expectedBiasSize, len(gradBias))
	}

	// Check that gradients are not all zeros
	allZero := true
	for _, v := range gradWeights.Data {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("Weight gradients are all zeros")
	}

	// Verify output is deterministic
	_ = output
}

func TestGRUParams(t *testing.T) {
	ResetRand(42)

	inputSize := 4
	hiddenSize := 3

	gru := NewGRULayer(inputSize, hiddenSize)

	weights, bias := gru.Params()

	// Expected sizes
	expectedWeightSize := 3*hiddenSize*inputSize + 3*hiddenSize*hiddenSize
	expectedBiasSize := hiddenSize * 3

	if len(weights.Data) != expectedWeightSize {
		t.Errorf("Expected weight size %d, got %d", expectedWeightSize, len(weights.Data))
	}

	if len(bias) != expectedBiasSize {
		t.Errorf("Expected bias size %d, got %d", expectedBiasSize, len(bias))
	}

	// Test SetParams
	gru2 := NewGRULayer(inputSize, hiddenSize)
	gru2.SetParams(weights, bias)

	weights2, bias2 := gru2.Params()

	for i := range weights.Data {
		if weights.Data[i] != weights2.Data[i] {
			t.Errorf("SetParams failed: weight mismatch at %d", i)
			break
		}
	}

	for i := range bias {
		if bias[i] != bias2[i] {
			t.Errorf("SetParams failed: bias mismatch at %d", i)
			break
		}
	}
}

func TestGRUType(t *testing.T) {
	gru := NewGRULayer(10, 5)
	if gru.Type() != "gru" {
		t.Errorf("Expected type 'gru', got '%s'", gru.Type())
	}
}
