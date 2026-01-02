package model

import (
	"testing"
)

func TestConv2D(t *testing.T) {
	t.Run("Identity Convolution", func(t *testing.T) {
		// 1 channel, 3x3 input
		input := NewTensor([]int{1, 3, 3})
		for i := range input.Data {
			input.Data[i] = float32(i + 1)
		}

		// 1 filter, 1x1 kernel, value 1.0 (Identity)
		weights := NewTensor([]int{1, 1, 1, 1})
		weights.Data[0] = 1.0
		bias := make([]float32, 1)

		output := Conv2D(input, weights, bias, 1, 0)

		if output.Shape[1] != 3 || output.Shape[2] != 3 {
			t.Errorf("Expected 3x3 output, got %dx%d", output.Shape[1], output.Shape[2])
		}

		for i := range output.Data {
			if output.Data[i] != input.Data[i] {
				t.Errorf("Expected %f, got %f at index %d", input.Data[i], output.Data[i], i)
			}
		}
	})

	t.Run("3x3 Kernel with Padding", func(t *testing.T) {
		input := NewTensor([]int{1, 3, 3})
		for i := range input.Data {
			input.Data[i] = 1.0
		}

		// 1 filter, 3x3 kernel, all values 1.0
		weights := NewTensor([]int{1, 1, 3, 3})
		for i := range weights.Data {
			weights.Data[i] = 1.0
		}
		bias := make([]float32, 1)

		// With padding=1, 3x3 input stays 3x3
		output := Conv2D(input, weights, bias, 1, 1)

		if output.Shape[1] != 3 || output.Shape[2] != 3 {
			t.Errorf("Expected 3x3 output, got %dx%d", output.Shape[1], output.Shape[2])
		}

		// Center value should be sum of 3x3 = 9.0
		centerVal := output.Get([]int{0, 1, 1})
		if centerVal != 9.0 {
			t.Errorf("Expected 9.0 at center, got %f", centerVal)
		}
	})
}
