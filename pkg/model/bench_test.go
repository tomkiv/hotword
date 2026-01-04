package model

import (
	"testing"
)

func BenchmarkConv2D(b *testing.B) {
	// Large input: [16, 64, 64]
	input := NewTensor([]int{16, 64, 64})
	for i := range input.Data {
		input.Data[i] = 0.1
	}
	
	// 32 filters, 3x3 kernel
	weights := NewTensor([]int{32, 16, 3, 3})
	for i := range weights.Data {
		weights.Data[i] = 0.1
	}
	bias := make([]float32, 32)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Conv2D(input, weights, bias, 1, 1)
	}
}

func BenchmarkMaxPool2D(b *testing.B) {
	// Large input: [32, 64, 64]
	input := NewTensor([]int{32, 64, 64})
	for i := range input.Data {
		input.Data[i] = 0.1
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MaxPool2D(input, 2, 2)
	}
}
