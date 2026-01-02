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
