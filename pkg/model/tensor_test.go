package model

import (
	"testing"
)

func TestTensor(t *testing.T) {
	t.Run("Create and Shape", func(t *testing.T) {
		shape := []int{2, 3, 4}
		tensor := NewTensor(shape)
		if len(tensor.Data) != 2*3*4 {
			t.Errorf("Expected data size 24, got %d", len(tensor.Data))
		}
		for i, v := range shape {
			if tensor.Shape[i] != v {
				t.Errorf("Expected shape[%d] = %d, got %d", i, v, tensor.Shape[i])
			}
		}
	})

	t.Run("Get and Set", func(t *testing.T) {
		tensor := NewTensor([]int{2, 2})
		tensor.Set([]int{1, 1}, 5.5)
		val := tensor.Get([]int{1, 1})
		if val != 5.5 {
			t.Errorf("Expected 5.5, got %f", val)
		}
	})
}
