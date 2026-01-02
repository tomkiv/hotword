package model

// Conv2D performs a 2D convolution operation.
// input: [channels, height, width]
// weights: [num_filters, input_channels, kernel_height, kernel_width]
// bias: [num_filters]
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

// MaxPool2D performs a 2D max pooling operation.
// input: [channels, height, width]
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
// input: any shape (will be flattened)
// weights: [output_units, input_size]
// bias: [output_units]
func Dense(input, weights *Tensor, bias []float32) *Tensor {
	numOutputs := weights.Shape[0]
	inputSize := len(input.Data)

	output := NewTensor([]int{numOutputs})

	for i := 0; i < numOutputs; i++ {
		var sum float32
		for j := 0; j < inputSize; j++ {
			sum += input.Data[j] * weights.Get([]int{i, j})
		}
		output.Data[i] = sum + bias[i]
	}

	return output
}

// DenseBackward calculates the gradients for the Dense layer.
// input: [input_size]
// weights: [num_outputs, input_size]
// bias: [num_outputs]
// gradOutput: [num_outputs]
// Returns: gradInput [input_size], gradWeights [num_outputs, input_size], gradBias [num_outputs]
func DenseBackward(input, weights *Tensor, bias []float32, gradOutput *Tensor) (*Tensor, *Tensor, []float32) {
	numOutputs := weights.Shape[0]
	inputSize := weights.Shape[1]

	gradInput := NewTensor(input.Shape)
	gradWeights := NewTensor(weights.Shape)
	gradBias := make([]float32, numOutputs)

	for i := 0; i < numOutputs; i++ {
		goi := gradOutput.Data[i]
		gradBias[i] = goi

		for j := 0; j < inputSize; j++ {
			// gradWeights = gradOutput * input^T
			gradWeights.Set([]int{i, j}, goi*input.Data[j])

			// gradInput = W^T * gradOutput
			gradInput.Data[j] += weights.Get([]int{i, j}) * goi
		}
	}

	return gradInput, gradWeights, gradBias
}
