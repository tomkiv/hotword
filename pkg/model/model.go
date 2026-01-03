package model

// Model represents a neural network model.
type Model interface {
	Forward(input *Tensor) *Tensor
	GetWeights() *Tensor
	GetBias() []float32
}

// DenseModel is a simple model with a single Dense layer and Sigmoid activation.
type DenseModel struct {
	weights *Tensor
	bias    []float32
}

// NewDenseModel creates a new DenseModel.
func NewDenseModel(weights *Tensor, bias []float32) *DenseModel {
	return &DenseModel{
		weights: weights,
		bias:    bias,
	}
}

// Forward performs the forward pass: Sigmoid(Dense(input)).
func (m *DenseModel) Forward(input *Tensor) *Tensor {
	logits := Dense(input, m.weights, m.bias)
	return Sigmoid(logits)
}

func (m *DenseModel) GetWeights() *Tensor {
	return m.weights
}

func (m *DenseModel) GetBias() []float32 {
	return m.bias
}
