package model

import (
	"testing"
)

func TestLSTMForward(t *testing.T) {
	ResetRand(42)
	inputSize, hiddenSize, seqLen := 10, 8, 5
	lstm := NewLSTMLayer(inputSize, hiddenSize)

	input := NewTensor([]int{seqLen, inputSize})
	for i := range input.Data {
		input.Data[i] = randFloat32()*2 - 1
	}

	output := lstm.Forward(input)
	if output == nil {
		t.Fatal("LSTM forward returned nil")
	}
	if len(output.Data) != hiddenSize {
		t.Errorf("Expected output size %d, got %d", hiddenSize, len(output.Data))
	}
}

func TestLSTMBackward(t *testing.T) {
	ResetRand(42)
	inputSize, hiddenSize, seqLen := 4, 3, 2
	lstm := NewLSTMLayer(inputSize, hiddenSize)

	input := NewTensor([]int{seqLen, inputSize})
	for i := range input.Data {
		input.Data[i] = randFloat32()*2 - 1
	}

	output := lstm.Forward(input)
	_ = output
	gradOutput := NewTensor([]int{hiddenSize})
	for i := range gradOutput.Data {
		gradOutput.Data[i] = 1.0
	}

	gradInput, gradWeights, gradBias := lstm.Backward(input, gradOutput)
	if gradInput == nil || gradWeights == nil || gradBias == nil {
		t.Fatal("Backward returned nil")
	}

	expectedWeightSize := 4*hiddenSize*inputSize + 4*hiddenSize*hiddenSize
	if len(gradWeights.Data) != expectedWeightSize {
		t.Errorf("Expected weight gradient size %d, got %d", expectedWeightSize, len(gradWeights.Data))
	}
}
