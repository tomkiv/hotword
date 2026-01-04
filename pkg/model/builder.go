package model

import (
	"math"
	"math/rand"
)

// LayerConfig defines the configuration for a single layer.
type LayerConfig struct {
	Type       string `mapstructure:"type"`
	Filters    int    `mapstructure:"filters"`
	KernelSize int    `mapstructure:"kernel"`
	Stride     int    `mapstructure:"stride"`
	Padding    int    `mapstructure:"padding"`
	Units      int    `mapstructure:"units"`
}

// BuildModelFromConfig constructs a SequentialModel from a list of layer configurations.
func BuildModelFromConfig(configs []LayerConfig, inputShape []int) (*SequentialModel, error) {
	var layers []Layer
	currentShape := inputShape

	for _, cfg := range configs {
		var layer Layer
		switch cfg.Type {
		case "conv2d":
			// weights: [num_filters, input_channels, kernel_height, kernel_width]
			inChannels := currentShape[0]
			weightsShape := []int{cfg.Filters, inChannels, cfg.KernelSize, cfg.KernelSize}
			weights := NewTensor(weightsShape)

			// Xavier initialization
			fanIn := inChannels * cfg.KernelSize * cfg.KernelSize
			fanOut := cfg.Filters * cfg.KernelSize * cfg.KernelSize
			scale := float32(math.Sqrt(6.0 / float64(fanIn+fanOut)))
			for i := range weights.Data {
				weights.Data[i] = (rand.Float32()*2 - 1) * scale
			}

			bias := make([]float32, cfg.Filters)
			layer = NewConv2DLayer(weights, bias, cfg.Stride, cfg.Padding)

			// Update shape
			outHeight := (currentShape[1]+2*cfg.Padding-cfg.KernelSize)/cfg.Stride + 1
			outWidth := (currentShape[2]+2*cfg.Padding-cfg.KernelSize)/cfg.Stride + 1
			currentShape = []int{cfg.Filters, outHeight, outWidth}

		case "relu":
			layer = NewReLULayer()
		case "sigmoid":
			layer = NewSigmoidLayer()
		case "maxpool2d":
			layer = NewMaxPool2DLayer(cfg.KernelSize, cfg.Stride)
			// Update shape
			outHeight := (currentShape[1]-cfg.KernelSize)/cfg.Stride + 1
			outWidth := (currentShape[2]-cfg.KernelSize)/cfg.Stride + 1
			currentShape = []int{currentShape[0], outHeight, outWidth}

		case "dense":
			inSize := 1
			for _, dim := range currentShape {
				inSize *= dim
			}
			weightsShape := []int{cfg.Units, inSize}
			weights := NewTensor(weightsShape)

			// Xavier initialization
			scale := float32(math.Sqrt(6.0 / float64(inSize+cfg.Units)))
			for i := range weights.Data {
				weights.Data[i] = (rand.Float32()*2 - 1) * scale
			}

			bias := make([]float32, cfg.Units)
			layer = NewDenseLayer(weights, bias)
			currentShape = []int{cfg.Units}

		case "gru", "lstm":
			// GRU/LSTM expect input from CNN: [channels, height, width]
			// They will reshape to [height, channels*width] internally
			var inputSize int
			if len(currentShape) == 3 {
				// From CNN: [channels, height, width]
				inputSize = currentShape[0] * currentShape[2] // channels * width
			} else if len(currentShape) == 1 {
				inputSize = currentShape[0]
			} else {
				return nil, ErrUnsupportedLayer{Type: cfg.Type + " (invalid input shape)"}
			}

			hiddenSize := cfg.Units
			if hiddenSize == 0 {
				hiddenSize = 32 // Default hidden size
			}

			if cfg.Type == "gru" {
				layer = NewGRULayer(inputSize, hiddenSize)
			} else {
				layer = NewLSTMLayer(inputSize, hiddenSize)
			}
			currentShape = []int{hiddenSize} // Output is just the final hidden state

		default:
			return nil, ErrUnsupportedLayer{Type: cfg.Type}
		}
		layers = append(layers, layer)
	}

	return NewSequentialModel(layers...), nil
}
