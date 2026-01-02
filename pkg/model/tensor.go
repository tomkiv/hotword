package model

// Tensor represents a multi-dimensional array of float32 values.
type Tensor struct {
	Data  []float32
	Shape []int
}

// NewTensor creates a new Tensor with the given shape.
func NewTensor(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return &Tensor{
		Data:  make([]float32, size),
		Shape: shape,
	}
}

// Get returns the value at the specified indices.
func (t *Tensor) Get(indices []int) float32 {
	return t.Data[t.getIndex(indices)]
}

// Set sets the value at the specified indices.
func (t *Tensor) Set(indices []int, value float32) {
	t.Data[t.getIndex(indices)] = value
}

// getIndex calculates the flat index for the given multi-dimensional indices.
func (t *Tensor) getIndex(indices []int) int {
	index := 0
	multiplier := 1
	for i := len(indices) - 1; i >= 0; i-- {
		index += indices[i] * multiplier
		multiplier *= t.Shape[i]
	}
	return index
}
