package model

// Layer represents a single neural network layer.
type Layer interface {
	Forward(input *Tensor) *Tensor
	// Backward returns gradInput, gradWeights, gradBias
	Backward(input, gradOutput *Tensor) (*Tensor, *Tensor, []float32)
	// Params returns the weights and biases of the layer (if any)
	Params() (*Tensor, []float32)
	// SetParams sets the weights and biases of the layer
	SetParams(weights *Tensor, bias []float32)
	// Type returns the layer type name
	Type() string
}

// Model represents a neural network model.
type Model interface {
	Forward(input *Tensor) *Tensor
	GetLayers() []Layer
}

// SequentialModel is a model consisting of a sequence of layers.
type SequentialModel struct {
	Layers []Layer
}

// NewSequentialModel creates a new SequentialModel.
func NewSequentialModel(layers ...Layer) *SequentialModel {
	return &SequentialModel{Layers: layers}
}

// Forward performs the forward pass through all layers.
func (m *SequentialModel) Forward(input *Tensor) *Tensor {
	out := input
	for _, layer := range m.Layers {
		out = layer.Forward(out)
	}
	return out
}

func (m *SequentialModel) GetLayers() []Layer {
	return m.Layers
}