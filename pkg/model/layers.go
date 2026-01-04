package model

import (
	"math"
)

// Conv2DLayer represents a 2D convolutional layer.
type Conv2DLayer struct {
	Weights *Tensor
	Bias    []float32
	Stride  int
	Padding int
}

func NewConv2DLayer(weights *Tensor, bias []float32, stride, padding int) *Conv2DLayer {
	return &Conv2DLayer{
		Weights: weights,
		Bias:    bias,
		Stride:  stride,
		Padding: padding,
	}
}

func (l *Conv2DLayer) Forward(input *Tensor) *Tensor {
	return Conv2D(input, l.Weights, l.Bias, l.Stride, l.Padding)
}

func (l *Conv2DLayer) ForwardStateful(input *Tensor) *Tensor {
	return l.Forward(input)
}

func (l *Conv2DLayer) ResetState() {}

func (l *Conv2DLayer) Backward(input, gradOutput *Tensor) (*Tensor, *Tensor, []float32) {
	return Conv2DBackward(input, l.Weights, l.Bias, gradOutput, l.Stride, l.Padding)
}

func (l *Conv2DLayer) Params() (*Tensor, []float32) {
	return l.Weights, l.Bias
}

func (l *Conv2DLayer) SetParams(weights *Tensor, bias []float32) {
	l.Weights = weights
	l.Bias = bias
}

func (l *Conv2DLayer) Type() string {
	return "conv2d"
}

// ReLULayer represents a Rectified Linear Unit activation layer.
type ReLULayer struct{}

func NewReLULayer() *ReLULayer {
	return &ReLULayer{}
}

func (l *ReLULayer) Forward(input *Tensor) *Tensor {
	return ReLU(input)
}

func (l *ReLULayer) ForwardStateful(input *Tensor) *Tensor {
	return l.Forward(input)
}

func (l *ReLULayer) ResetState() {}

func (l *ReLULayer) Backward(input, gradOutput *Tensor) (*Tensor, *Tensor, []float32) {
	return ReLUBackward(input, gradOutput), nil, nil
}

func (l *ReLULayer) Params() (*Tensor, []float32) {
	return nil, nil
}

func (l *ReLULayer) SetParams(weights *Tensor, bias []float32) {}

func (l *ReLULayer) Type() string {
	return "relu"
}

// SigmoidLayer represents a Sigmoid activation layer.
type SigmoidLayer struct{}

func NewSigmoidLayer() *SigmoidLayer {
	return &SigmoidLayer{}
}

func (l *SigmoidLayer) Forward(input *Tensor) *Tensor {
	return Sigmoid(input)
}

func (l *SigmoidLayer) ForwardStateful(input *Tensor) *Tensor {
	return l.Forward(input)
}

func (l *SigmoidLayer) ResetState() {}

func (l *SigmoidLayer) Backward(input, gradOutput *Tensor) (*Tensor, *Tensor, []float32) {
	// gradInput = gradOutput * Sigmoid(input) * (1 - Sigmoid(input))
	out := Sigmoid(input)
	gradInput := NewTensor(input.Shape)
	for i := range gradInput.Data {
		gradInput.Data[i] = gradOutput.Data[i] * out.Data[i] * (1.0 - out.Data[i])
	}
	return gradInput, nil, nil
}

func (l *SigmoidLayer) Params() (*Tensor, []float32) {
	return nil, nil
}

func (l *SigmoidLayer) SetParams(weights *Tensor, bias []float32) {}

func (l *SigmoidLayer) Type() string {
	return "sigmoid"
}

// MaxPool2DLayer represents a 2D max pooling layer.
type MaxPool2DLayer struct {
	KernelSize int
	Stride     int
}

func NewMaxPool2DLayer(kernelSize, stride int) *MaxPool2DLayer {
	return &MaxPool2DLayer{
		KernelSize: kernelSize,
		Stride:     stride,
	}
}

func (l *MaxPool2DLayer) Forward(input *Tensor) *Tensor {
	return MaxPool2D(input, l.KernelSize, l.Stride)
}

func (l *MaxPool2DLayer) ForwardStateful(input *Tensor) *Tensor {
	return l.Forward(input)
}

func (l *MaxPool2DLayer) ResetState() {}

func (l *MaxPool2DLayer) Backward(input, gradOutput *Tensor) (*Tensor, *Tensor, []float32) {
	return MaxPool2DBackward(input, gradOutput, l.KernelSize, l.Stride), nil, nil
}

func (l *MaxPool2DLayer) Params() (*Tensor, []float32) {
	return nil, nil
}

func (l *MaxPool2DLayer) SetParams(weights *Tensor, bias []float32) {}

func (l *MaxPool2DLayer) Type() string {
	return "maxpool2d"
}

// DenseLayer represents a fully connected layer.
type DenseLayer struct {
	Weights *Tensor
	Bias    []float32
}

func NewDenseLayer(weights *Tensor, bias []float32) *DenseLayer {
	return &DenseLayer{
		Weights: weights,
		Bias:    bias,
	}
}

func (l *DenseLayer) Forward(input *Tensor) *Tensor {
	return Dense(input, l.Weights, l.Bias)
}

func (l *DenseLayer) ForwardStateful(input *Tensor) *Tensor {
	return l.Forward(input)
}

func (l *DenseLayer) ResetState() {}

func (l *DenseLayer) Backward(input, gradOutput *Tensor) (*Tensor, *Tensor, []float32) {
	return DenseBackward(input, l.Weights, l.Bias, gradOutput)
}

func (l *DenseLayer) Params() (*Tensor, []float32) {
	return l.Weights, l.Bias
}

func (l *DenseLayer) SetParams(weights *Tensor, bias []float32) {
	l.Weights = weights
	l.Bias = bias
}

func (l *DenseLayer) Type() string {
	return "dense"
}

// --- Functional Helpers ---

// Conv2D performs a 2D convolution operation.
func Conv2D(input, weights *Tensor, bias []float32, stride, padding int) *Tensor {
	inChannels := input.Shape[0]
	inHeight := input.Shape[1]
	inWidth := input.Shape[2]

	numFilters := weights.Shape[0]
	kernelHeight := weights.Shape[2]	
	kernelWidth := weights.Shape[3]

	outHeight := (inHeight+2*padding-kernelHeight)/stride + 1
	outWidth := (inWidth+2*padding-kernelWidth)/stride + 1

	output := NewTensor([]int{numFilters, outHeight, outWidth})

	for f := 0; f < numFilters; f++ {
		for i := 0; i < outHeight; i++ {
			for j := 0; j < outWidth; j++ {
				var sum float32
				for c := 0; c < inChannels; c++ {
					for ki := 0; ki < kernelHeight; ki++ {
						for kj := 0; kj < kernelWidth; kj++ {
							ii := i*stride - padding + ki
							jj := j*stride - padding + kj

							if ii >= 0 && ii < inHeight && jj >= 0 && jj < inWidth {
								val := input.Get([]int{c, ii, jj})
								weight := weights.Get([]int{f, c, ki, kj})
								sum += val * weight
							}
						}
					}
				}
				output.Set([]int{f, i, j}, sum+bias[f])
			}
		}
	}

	return output
}

// ReLU applies the rectified linear unit activation function.
func ReLU(input *Tensor) *Tensor {
	output := NewTensor(input.Shape)
	for i, val := range input.Data {
		if val > 0 {
			output.Data[i] = val
		} else {
			output.Data[i] = 0
		}
	}
	return output
}

// Sigmoid applies the sigmoid activation function to the input tensor.
func Sigmoid(input *Tensor) *Tensor {
	output := NewTensor(input.Shape)
	for i, val := range input.Data {
		output.Data[i] = 1.0 / (1.0 + float32(math.Exp(float64(-val))))
	}
	return output
}

// MaxPool2D performs a 2D max pooling operation.
func MaxPool2D(input *Tensor, kernelSize, stride int) *Tensor {
	channels := input.Shape[0]
	inHeight := input.Shape[1]
	inWidth := input.Shape[2]

	outHeight := (inHeight-kernelSize)/stride + 1
	outWidth := (inWidth-kernelSize)/stride + 1

	output := NewTensor([]int{channels, outHeight, outWidth})

	for c := 0; c < channels; c++ {
		for i := 0; i < outHeight; i++ {
			for j := 0; j < outWidth; j++ {
				var maxVal float32 = -3.402823466e+38 // float32 min
				
				for ki := 0; ki < kernelSize; ki++ {
					for kj := 0; kj < kernelSize; kj++ {
						ii := i*stride + ki
						jj := j*stride + kj
						
						val := input.Get([]int{c, ii, jj})
						if val > maxVal {
							maxVal = val
						}
					}
				}
				output.Set([]int{c, i, j}, maxVal)
			}
		}
	}

	return output
}

// Dense performs a fully connected (dense) layer operation.
func Dense(input, weights *Tensor, bias []float32) *Tensor {
	numOutputs := weights.Shape[0]
	expectedInputSize := weights.Shape[1]
	actualInputSize := len(input.Data)

	if actualInputSize != expectedInputSize {
		panic("Dense layer: input size mismatch")
	}

	output := NewTensor([]int{numOutputs})

	for i := 0; i < numOutputs; i++ {
		var sum float32
		for j := 0; j < expectedInputSize; j++ {
			sum += input.Data[j] * weights.Get([]int{i, j})
		}
		output.Data[i] = sum + bias[i]
	}

	return output
}

// DenseBackward calculates the gradients for the Dense layer.
func DenseBackward(input, weights *Tensor, bias []float32, gradOutput *Tensor) (*Tensor, *Tensor, []float32) {
	numOutputs := weights.Shape[0]
	expectedInputSize := weights.Shape[1]
	actualInputSize := len(input.Data)

	if actualInputSize != expectedInputSize {
		panic("DenseBackward: input size mismatch")
	}

	gradInput := NewTensor(input.Shape)
	gradWeights := NewTensor(weights.Shape)
	gradBias := make([]float32, numOutputs)

	for i := 0; i < numOutputs; i++ {
		goi := gradOutput.Data[i]
		gradBias[i] = goi

		for j := 0; j < expectedInputSize; j++ {
			gradWeights.Set([]int{i, j}, goi*input.Data[j])
			gradInput.Data[j] += weights.Get([]int{i, j}) * goi
		}
	}

	return gradInput, gradWeights, gradBias
}

// ReLUBackward calculates the gradient for the ReLU activation.
func ReLUBackward(input, gradOutput *Tensor) *Tensor {
	gradInput := NewTensor(input.Shape)
	for i, val := range input.Data {
		if val > 0 {
			gradInput.Data[i] = gradOutput.Data[i]
		} else {
			gradInput.Data[i] = 0
		}
	}
	return gradInput
}

// Conv2DBackward calculates the gradients for the Conv2D layer.
func Conv2DBackward(input, weights *Tensor, bias []float32, gradOutput *Tensor, stride, padding int) (*Tensor, *Tensor, []float32) {
	inChannels := input.Shape[0]
	inHeight := input.Shape[1]
	inWidth := input.Shape[2]

	numFilters := weights.Shape[0]
	kernelHeight := weights.Shape[2]
	kernelWidth := weights.Shape[3]

	outHeight := gradOutput.Shape[1]
	outWidth := gradOutput.Shape[2]

	gradInput := NewTensor(input.Shape)
	gradWeights := NewTensor(weights.Shape)
	gradBias := make([]float32, numFilters)

	for f := 0; f < numFilters; f++ {
		for i := 0; i < outHeight; i++ {
			for j := 0; j < outWidth; j++ {
				goVal := gradOutput.Get([]int{f, i, j})
				gradBias[f] += goVal

				for c := 0; c < inChannels; c++ {
					for ki := 0; ki < kernelHeight; ki++ {
						for kj := 0; kj < kernelWidth; kj++ {
							ii := i*stride - padding + ki
							jj := j*stride - padding + kj

							if ii >= 0 && ii < inHeight && jj >= 0 && jj < inWidth {
								inVal := input.Get([]int{c, ii, jj})
								gradWeights.Data[gradWeights.getIndex([]int{f, c, ki, kj})] += inVal * goVal

								wVal := weights.Get([]int{f, c, ki, kj})
								gradInput.Data[gradInput.getIndex([]int{c, ii, jj})] += wVal * goVal
							}
						}
					}
				}
			}
		}
	}

	return gradInput, gradWeights, gradBias
}

// MaxPool2DBackward calculates the gradient for the MaxPool2D layer.
func MaxPool2DBackward(input, gradOutput *Tensor, kernelSize, stride int) *Tensor {
	channels := input.Shape[0]
	outHeight := gradOutput.Shape[1]
	outWidth := gradOutput.Shape[2]
	gradInput := NewTensor(input.Shape)

	for c := 0; c < channels; c++ {
		for i := 0; i < outHeight; i++ {
			for j := 0; j < outWidth; j++ {
				var maxVal float32 = -3.402823466e+38
				maxIdx := []int{0, 0}

				for ki := 0; ki < kernelSize; ki++ {
					for kj := 0; kj < kernelSize; kj++ {
						ii := i*stride + ki
						jj := j*stride + kj

						val := input.Get([]int{c, ii, jj})
						if val > maxVal {
							maxVal = val
							maxIdx[0] = ii
							maxIdx[1] = jj
						}
					}
				}
				
				flatIdx := gradInput.getIndex([]int{c, maxIdx[0], maxIdx[1]})
				gradInput.Data[flatIdx] += gradOutput.Get([]int{c, i, j})
			}
		}
	}

	return gradInput
}