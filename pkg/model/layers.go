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
				var maxVal float32 = -3.402823466e+38 // math.SmallestNonzeroFloat32 is positive, using float32 min
				
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
