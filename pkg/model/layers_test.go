package model

import (
	"testing"
)

func TestActivationAndPooling(t *testing.T) {
	t.Run("ReLU", func(t *testing.T) {
		input := NewTensor([]int{1, 2})
		input.Data[0] = -1.0
		input.Data[1] = 2.0
		
		output := ReLU(input)
		if output.Data[0] != 0.0 {
			t.Errorf("Expected 0.0, got %f", output.Data[0])
		}
		if output.Data[1] != 2.0 {
			t.Errorf("Expected 2.0, got %f", output.Data[1])
		}
	})

	t.Run("MaxPool2D", func(t *testing.T) {
		input := NewTensor([]int{1, 4, 4})
		// Set a max value in one quadrant
		input.Set([]int{0, 0, 0}, 10.0)
		input.Set([]int{0, 2, 2}, 20.0)

		output := MaxPool2D(input, 2, 2)

		if output.Shape[1] != 2 || output.Shape[2] != 2 {
			t.Errorf("Expected 2x2 output, got %dx%d", output.Shape[1], output.Shape[2])
		}

		if output.Get([]int{0, 0, 0}) != 10.0 {
			t.Errorf("Expected 10.0, got %f", output.Get([]int{0, 0, 0}))
		}
		if output.Get([]int{0, 1, 1}) != 20.0 {
			t.Errorf("Expected 20.0, got %f", output.Get([]int{0, 1, 1}))
		}
	})
}

func TestDense(t *testing.T) {
	t.Run("Basic Dense Layer", func(t *testing.T) {
		// Flattened input of size 4
		input := NewTensor([]int{4})
		for i := range input.Data {
			input.Data[i] = float32(i + 1) // 1, 2, 3, 4
		}

		// Weights for 2 output units (2x4 matrix)
		weights := NewTensor([]int{2, 4})
		for i := range weights.Data {
			weights.Data[i] = 1.0
		}
		bias := []float32{0.5, -0.5}

		output := Dense(input, weights, bias)

		if output.Shape[0] != 2 {
			t.Errorf("Expected output size 2, got %d", output.Shape[0])
		}

		// Unit 1: (1*1 + 2*1 + 3*1 + 4*1) + 0.5 = 10.5
		if output.Data[0] != 10.5 {
			t.Errorf("Expected 10.5, got %f", output.Data[0])
		}
		// Unit 2: (1*1 + 2*1 + 3*1 + 4*1) - 0.5 = 9.5
		if output.Data[1] != 9.5 {
			t.Errorf("Expected 9.5, got %f", output.Data[1])
		}
	})

	t.Run("Flattened Input Dense", func(t *testing.T) {
		// 2x2 input (should be treated as flat size 4)
		input := NewTensor([]int{1, 2, 2})
		for i := range input.Data {
			input.Data[i] = 1.0
		}

		weights := NewTensor([]int{1, 4})
		for i := range weights.Data {
			weights.Data[i] = 1.0
		}
		bias := []float32{0.0}

		output := Dense(input, weights, bias)
		if output.Data[0] != 4.0 {
			t.Errorf("Expected 4.0, got %f", output.Data[0])
		}
	})
}
