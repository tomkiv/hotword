package model

import (
	"math"
)

// GRULayer implements a Gated Recurrent Unit layer for sequence processing.
// It processes input sequences and learns temporal dependencies.
type GRULayer struct {
	// Input weights [3, hiddenSize, inputSize] for z, r, h gates
	Wz, Wr, Wh *Tensor
	// Hidden weights [3, hiddenSize, hiddenSize] for z, r, h gates
	Uz, Ur, Uh *Tensor
	// Biases [hiddenSize] for each gate
	Bz, Br, Bh []float32

	InputSize  int
	HiddenSize int

	// Stateful inference
	hiddenState []float32

	// Cached values for backward pass
	lastInput          *Tensor
	originalInputShape []int     // Store original 3D shape for reshaping gradient
	hiddenSeq          []*Tensor // Hidden states at each timestep
	zSeq               []*Tensor // Update gate values
	rSeq               []*Tensor // Reset gate values
	hCandSeq           []*Tensor // Candidate hidden values
}

// NewGRULayer creates a new GRU layer with the specified dimensions.
func NewGRULayer(inputSize, hiddenSize int) *GRULayer {
	// Initialize weights with Xavier initialization
	scale := float32(math.Sqrt(6.0 / float64(inputSize+hiddenSize)))

	wz := NewTensor([]int{hiddenSize, inputSize})
	wr := NewTensor([]int{hiddenSize, inputSize})
	wh := NewTensor([]int{hiddenSize, inputSize})

	uz := NewTensor([]int{hiddenSize, hiddenSize})
	ur := NewTensor([]int{hiddenSize, hiddenSize})
	uh := NewTensor([]int{hiddenSize, hiddenSize})

	// Xavier initialization for all weight matrices
	for i := range wz.Data {
		wz.Data[i] = (randFloat32()*2 - 1) * scale
		wr.Data[i] = (randFloat32()*2 - 1) * scale
		wh.Data[i] = (randFloat32()*2 - 1) * scale
	}

	hiddenScale := float32(math.Sqrt(6.0 / float64(2*hiddenSize)))
	for i := range uz.Data {
		uz.Data[i] = (randFloat32()*2 - 1) * hiddenScale
		ur.Data[i] = (randFloat32()*2 - 1) * hiddenScale
		uh.Data[i] = (randFloat32()*2 - 1) * hiddenScale
	}

	return &GRULayer{
		Wz:         wz,
		Wr:         wr,
		Wh:         wh,
		Uz:         uz,
		Ur:         ur,
		Uh:         uh,
		Bz:         make([]float32, hiddenSize),
		Br:         make([]float32, hiddenSize),
		Bh:         make([]float32, hiddenSize),
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
	}
}

// Forward processes a sequence through the GRU.
// Input shape: [seqLen, inputSize]
// Output: final hidden state [hiddenSize]
func (g *GRULayer) Forward(input *Tensor) *Tensor {
	return g.forwardInternal(input, nil)
}

// ForwardStateful processes a sequence maintaining internal hidden state.
func (g *GRULayer) ForwardStateful(input *Tensor) *Tensor {
	if g.hiddenState == nil {
		g.hiddenState = make([]float32, g.HiddenSize)
	}
	out := g.forwardInternal(input, g.hiddenState)
	copy(g.hiddenState, out.Data)
	return out
}

// ResetState clears the recurrent hidden state.
func (g *GRULayer) ResetState() {
	g.hiddenState = nil
}

func (g *GRULayer) forwardInternal(input *Tensor, initialH []float32) *Tensor {
	// Flatten input if it's 3D (from CNN output)
	var flatInput *Tensor
	var seqLen int

	if len(input.Shape) == 3 {
		// Input is [channels, height, width] - treat height as sequence length
		g.originalInputShape = input.Shape // Store for backward
		channels := input.Shape[0]
		height := input.Shape[1]
		width := input.Shape[2]
		seqLen = height
		featureSize := channels * width

		flatInput = NewTensor([]int{seqLen, featureSize})
		// Reshape: each time step gets all channel/width features
		for t := 0; t < seqLen; t++ {
			for c := 0; c < channels; c++ {
				for w := 0; w < width; w++ {
					srcIdx := c*height*width + t*width + w
					dstIdx := t*featureSize + c*width + w
					flatInput.Data[dstIdx] = input.Data[srcIdx]
				}
			}
		}
	} else if len(input.Shape) == 2 {
		g.originalInputShape = nil // No reshaping needed
		flatInput = input
		seqLen = input.Shape[0]
	} else {
		// Unsupported shape
		return nil
	}

	g.lastInput = flatInput

	// Initialize hidden state
	h := make([]float32, g.HiddenSize)
	if initialH != nil {
		copy(h, initialH)
	}

	// Store sequences for backward pass
	g.hiddenSeq = make([]*Tensor, seqLen+1)
	g.zSeq = make([]*Tensor, seqLen)
	g.rSeq = make([]*Tensor, seqLen)
	g.hCandSeq = make([]*Tensor, seqLen)

	// Initial hidden state
	g.hiddenSeq[0] = &Tensor{Data: make([]float32, g.HiddenSize), Shape: []int{g.HiddenSize}}
	copy(g.hiddenSeq[0].Data, h)

	inputSize := flatInput.Shape[1]

	// Process sequence
	for t := 0; t < seqLen; t++ {
		// Get input at timestep t
		xt := flatInput.Data[t*inputSize : (t+1)*inputSize]

		// Update gate: z = sigmoid(Wz*x + Uz*h + bz)
		z := make([]float32, g.HiddenSize)
		for i := 0; i < g.HiddenSize; i++ {
			sum := g.Bz[i]
			for j := 0; j < inputSize; j++ {
				sum += g.Wz.Data[i*inputSize+j] * xt[j]
			}
			for j := 0; j < g.HiddenSize; j++ {
				sum += g.Uz.Data[i*g.HiddenSize+j] * h[j]
			}
			z[i] = sigmoid32(sum)
		}

		// Reset gate: r = sigmoid(Wr*x + Ur*h + br)
		r := make([]float32, g.HiddenSize)
		for i := 0; i < g.HiddenSize; i++ {
			sum := g.Br[i]
			for j := 0; j < inputSize; j++ {
				sum += g.Wr.Data[i*inputSize+j] * xt[j]
			}
			for j := 0; j < g.HiddenSize; j++ {
				sum += g.Ur.Data[i*g.HiddenSize+j] * h[j]
			}
			r[i] = sigmoid32(sum)
		}

		// Candidate hidden: h_cand = tanh(Wh*x + Uh*(r*h) + bh)
		hCand := make([]float32, g.HiddenSize)
		for i := 0; i < g.HiddenSize; i++ {
			sum := g.Bh[i]
			for j := 0; j < inputSize; j++ {
				sum += g.Wh.Data[i*inputSize+j] * xt[j]
			}
			for j := 0; j < g.HiddenSize; j++ {
				sum += g.Uh.Data[i*g.HiddenSize+j] * (r[j] * h[j])
			}
			hCand[i] = tanh32(sum)
		}

		// New hidden state: h = (1-z)*h + z*h_cand
		newH := make([]float32, g.HiddenSize)
		for i := 0; i < g.HiddenSize; i++ {
			newH[i] = (1-z[i])*h[i] + z[i]*hCand[i]
		}

		// Store for backward pass
		g.zSeq[t] = &Tensor{Data: z, Shape: []int{g.HiddenSize}}
		g.rSeq[t] = &Tensor{Data: r, Shape: []int{g.HiddenSize}}
		g.hCandSeq[t] = &Tensor{Data: hCand, Shape: []int{g.HiddenSize}}
		g.hiddenSeq[t+1] = &Tensor{Data: newH, Shape: []int{g.HiddenSize}}

		h = newH
	}

	// Return final hidden state
	return &Tensor{Data: h, Shape: []int{g.HiddenSize}}
}

// ForwardWithMask processes a sequence through the GRU up to actualSeqLen timesteps.
// This supports variable-length inputs by stopping at the actual sequence length
// and returning the hidden state at that position (ignoring padded positions).
// Input shape: [seqLen, inputSize] (same as Forward)
// actualSeqLen: the actual number of valid timesteps (before padding)
// Output: hidden state at timestep actualSeqLen-1 [hiddenSize]
func (g *GRULayer) ForwardWithMask(input *Tensor, actualSeqLen int) *Tensor {
	// Flatten input if it's 3D (from CNN output)
	var flatInput *Tensor
	var seqLen int

	if len(input.Shape) == 3 {
		// Input is [channels, height, width] - treat height as sequence length
		g.originalInputShape = input.Shape
		channels := input.Shape[0]
		height := input.Shape[1]
		width := input.Shape[2]
		seqLen = height
		featureSize := channels * width

		flatInput = NewTensor([]int{seqLen, featureSize})
		for t := 0; t < seqLen; t++ {
			for c := 0; c < channels; c++ {
				for w := 0; w < width; w++ {
					srcIdx := c*height*width + t*width + w
					dstIdx := t*featureSize + c*width + w
					flatInput.Data[dstIdx] = input.Data[srcIdx]
				}
			}
		}
	} else if len(input.Shape) == 2 {
		g.originalInputShape = nil
		flatInput = input
		seqLen = input.Shape[0]
	} else {
		return nil
	}

	// Clamp actualSeqLen to valid range
	if actualSeqLen <= 0 {
		actualSeqLen = 1
	}
	if actualSeqLen > seqLen {
		actualSeqLen = seqLen
	}

	g.lastInput = flatInput

	// Initialize hidden state
	h := make([]float32, g.HiddenSize)

	// Store sequences for backward pass (only up to actualSeqLen)
	g.hiddenSeq = make([]*Tensor, actualSeqLen+1)
	g.zSeq = make([]*Tensor, actualSeqLen)
	g.rSeq = make([]*Tensor, actualSeqLen)
	g.hCandSeq = make([]*Tensor, actualSeqLen)

	g.hiddenSeq[0] = &Tensor{Data: make([]float32, g.HiddenSize), Shape: []int{g.HiddenSize}}
	copy(g.hiddenSeq[0].Data, h)

	inputSize := flatInput.Shape[1]

	// Process sequence only up to actualSeqLen (ignore padding)
	for t := 0; t < actualSeqLen; t++ {
		xt := flatInput.Data[t*inputSize : (t+1)*inputSize]

		// Update gate: z = sigmoid(Wz*x + Uz*h + bz)
		z := make([]float32, g.HiddenSize)
		for i := 0; i < g.HiddenSize; i++ {
			sum := g.Bz[i]
			for j := 0; j < inputSize; j++ {
				sum += g.Wz.Data[i*inputSize+j] * xt[j]
			}
			for j := 0; j < g.HiddenSize; j++ {
				sum += g.Uz.Data[i*g.HiddenSize+j] * h[j]
			}
			z[i] = sigmoid32(sum)
		}

		// Reset gate: r = sigmoid(Wr*x + Ur*h + br)
		r := make([]float32, g.HiddenSize)
		for i := 0; i < g.HiddenSize; i++ {
			sum := g.Br[i]
			for j := 0; j < inputSize; j++ {
				sum += g.Wr.Data[i*inputSize+j] * xt[j]
			}
			for j := 0; j < g.HiddenSize; j++ {
				sum += g.Ur.Data[i*g.HiddenSize+j] * h[j]
			}
			r[i] = sigmoid32(sum)
		}

		// Candidate hidden: h_cand = tanh(Wh*x + Uh*(r*h) + bh)
		hCand := make([]float32, g.HiddenSize)
		for i := 0; i < g.HiddenSize; i++ {
			sum := g.Bh[i]
			for j := 0; j < inputSize; j++ {
				sum += g.Wh.Data[i*inputSize+j] * xt[j]
			}
			for j := 0; j < g.HiddenSize; j++ {
				sum += g.Uh.Data[i*g.HiddenSize+j] * (r[j] * h[j])
			}
			hCand[i] = tanh32(sum)
		}

		// New hidden state: h = (1-z)*h + z*h_cand
		newH := make([]float32, g.HiddenSize)
		for i := 0; i < g.HiddenSize; i++ {
			newH[i] = (1-z[i])*h[i] + z[i]*hCand[i]
		}

		// Store for backward pass
		g.zSeq[t] = &Tensor{Data: z, Shape: []int{g.HiddenSize}}
		g.rSeq[t] = &Tensor{Data: r, Shape: []int{g.HiddenSize}}
		g.hCandSeq[t] = &Tensor{Data: hCand, Shape: []int{g.HiddenSize}}
		g.hiddenSeq[t+1] = &Tensor{Data: newH, Shape: []int{g.HiddenSize}}

		h = newH
	}

	// Return hidden state at actualSeqLen (not the padded end)
	return &Tensor{Data: h, Shape: []int{g.HiddenSize}}
}

// Backward computes gradients for the GRU layer using BPTT.
func (g *GRULayer) Backward(input, gradOutput *Tensor) (*Tensor, *Tensor, []float32) {
	seqLen := len(g.zSeq)
	inputSize := g.lastInput.Shape[1]

	// Initialize gradient accumulators for weights
	dWz := NewTensor(g.Wz.Shape)
	dWr := NewTensor(g.Wr.Shape)
	dWh := NewTensor(g.Wh.Shape)
	dUz := NewTensor(g.Uz.Shape)
	dUr := NewTensor(g.Ur.Shape)
	dUh := NewTensor(g.Uh.Shape)
	dBz := make([]float32, g.HiddenSize)
	dBr := make([]float32, g.HiddenSize)
	dBh := make([]float32, g.HiddenSize)

	// Gradient of hidden state (starts from output gradient)
	dh := make([]float32, g.HiddenSize)
	copy(dh, gradOutput.Data)

	// Gradient w.r.t. input
	dInput := NewTensor(g.lastInput.Shape)

	// Backprop through time
	for t := seqLen - 1; t >= 0; t-- {
		xt := g.lastInput.Data[t*inputSize : (t+1)*inputSize]
		hPrev := g.hiddenSeq[t].Data
		z := g.zSeq[t].Data
		r := g.rSeq[t].Data
		hCand := g.hCandSeq[t].Data

		// Gradient of update gate
		dz := make([]float32, g.HiddenSize)
		for i := 0; i < g.HiddenSize; i++ {
			dz[i] = dh[i] * (hCand[i] - hPrev[i]) * z[i] * (1 - z[i])
		}

		// Gradient of candidate hidden
		dhCand := make([]float32, g.HiddenSize)
		for i := 0; i < g.HiddenSize; i++ {
			dhCand[i] = dh[i] * z[i] * (1 - hCand[i]*hCand[i])
		}

		// Gradient of reset gate
		dr := make([]float32, g.HiddenSize)
		for i := 0; i < g.HiddenSize; i++ {
			var urh float32
			for j := 0; j < g.HiddenSize; j++ {
				urh += g.Uh.Data[i*g.HiddenSize+j] * hPrev[j]
			}
			dr[i] = dhCand[i] * urh * r[i] * (1 - r[i])
		}

		// Accumulate weight gradients
		for i := 0; i < g.HiddenSize; i++ {
			for j := 0; j < inputSize; j++ {
				dWz.Data[i*inputSize+j] += dz[i] * xt[j]
				dWr.Data[i*inputSize+j] += dr[i] * xt[j]
				dWh.Data[i*inputSize+j] += dhCand[i] * xt[j]
			}
			for j := 0; j < g.HiddenSize; j++ {
				dUz.Data[i*g.HiddenSize+j] += dz[i] * hPrev[j]
				dUr.Data[i*g.HiddenSize+j] += dr[i] * hPrev[j]
				dUh.Data[i*g.HiddenSize+j] += dhCand[i] * r[j] * hPrev[j]
			}
			dBz[i] += dz[i]
			dBr[i] += dr[i]
			dBh[i] += dhCand[i]
		}

		// Gradient w.r.t. input at timestep t
		for j := 0; j < inputSize; j++ {
			for i := 0; i < g.HiddenSize; i++ {
				dInput.Data[t*inputSize+j] += dz[i]*g.Wz.Data[i*inputSize+j] +
					dr[i]*g.Wr.Data[i*inputSize+j] +
					dhCand[i]*g.Wh.Data[i*inputSize+j]
			}
		}

		// Gradient w.r.t. previous hidden state
		newDh := make([]float32, g.HiddenSize)
		for j := 0; j < g.HiddenSize; j++ {
			newDh[j] = dh[j] * (1 - z[j])
			for i := 0; i < g.HiddenSize; i++ {
				newDh[j] += dz[i]*g.Uz.Data[i*g.HiddenSize+j] +
					dr[i]*g.Ur.Data[i*g.HiddenSize+j] +
					dhCand[i]*g.Uh.Data[i*g.HiddenSize+j]*r[j]
			}
		}
		dh = newDh
	}

	// Pack all weight gradients into a single tensor for the interface
	// We need to return combined gradients - pack Wz, Wr, Wh, Uz, Ur, Uh together
	totalWeightSize := len(dWz.Data) + len(dWr.Data) + len(dWh.Data) +
		len(dUz.Data) + len(dUr.Data) + len(dUh.Data)
	combinedGradWeights := NewTensor([]int{totalWeightSize})

	idx := 0
	copy(combinedGradWeights.Data[idx:], dWz.Data)
	idx += len(dWz.Data)
	copy(combinedGradWeights.Data[idx:], dWr.Data)
	idx += len(dWr.Data)
	copy(combinedGradWeights.Data[idx:], dWh.Data)
	idx += len(dWh.Data)
	copy(combinedGradWeights.Data[idx:], dUz.Data)
	idx += len(dUz.Data)
	copy(combinedGradWeights.Data[idx:], dUr.Data)
	idx += len(dUr.Data)
	copy(combinedGradWeights.Data[idx:], dUh.Data)

	// Combine bias gradients
	combinedGradBias := make([]float32, g.HiddenSize*3)
	copy(combinedGradBias[0:], dBz)
	copy(combinedGradBias[g.HiddenSize:], dBr)
	copy(combinedGradBias[g.HiddenSize*2:], dBh)

	// If original input was 3D, reshape gradient back to 3D
	var finalGradInput *Tensor
	if g.originalInputShape != nil && len(g.originalInputShape) == 3 {
		channels := g.originalInputShape[0]
		height := g.originalInputShape[1]
		width := g.originalInputShape[2]
		finalGradInput = NewTensor(g.originalInputShape)

		// Reverse the reshape: [seqLen, featureSize] -> [channels, height, width]
		featureSize := channels * width
		for t := 0; t < height; t++ {
			for c := 0; c < channels; c++ {
				for w := 0; w < width; w++ {
					srcIdx := t*featureSize + c*width + w
					dstIdx := c*height*width + t*width + w
					finalGradInput.Data[dstIdx] = dInput.Data[srcIdx]
				}
			}
		}
	} else {
		finalGradInput = dInput
	}

	return finalGradInput, combinedGradWeights, combinedGradBias
}

// Params returns the combined weights and biases.
func (g *GRULayer) Params() (*Tensor, []float32) {
	// Combine all weight tensors
	totalWeightSize := len(g.Wz.Data) + len(g.Wr.Data) + len(g.Wh.Data) +
		len(g.Uz.Data) + len(g.Ur.Data) + len(g.Uh.Data)
	combinedWeights := NewTensor([]int{totalWeightSize})

	idx := 0
	copy(combinedWeights.Data[idx:], g.Wz.Data)
	idx += len(g.Wz.Data)
	copy(combinedWeights.Data[idx:], g.Wr.Data)
	idx += len(g.Wr.Data)
	copy(combinedWeights.Data[idx:], g.Wh.Data)
	idx += len(g.Wh.Data)
	copy(combinedWeights.Data[idx:], g.Uz.Data)
	idx += len(g.Uz.Data)
	copy(combinedWeights.Data[idx:], g.Ur.Data)
	idx += len(g.Ur.Data)
	copy(combinedWeights.Data[idx:], g.Uh.Data)

	// Combine biases
	combinedBias := make([]float32, g.HiddenSize*3)
	copy(combinedBias[0:], g.Bz)
	copy(combinedBias[g.HiddenSize:], g.Br)
	copy(combinedBias[g.HiddenSize*2:], g.Bh)

	return combinedWeights, combinedBias
}

// SetParams updates the weights and biases from combined tensors.
func (g *GRULayer) SetParams(weights *Tensor, bias []float32) {
	inputSize := g.InputSize
	hiddenSize := g.HiddenSize
	wSize := hiddenSize * inputSize
	uSize := hiddenSize * hiddenSize

	idx := 0
	copy(g.Wz.Data, weights.Data[idx:idx+wSize])
	idx += wSize
	copy(g.Wr.Data, weights.Data[idx:idx+wSize])
	idx += wSize
	copy(g.Wh.Data, weights.Data[idx:idx+wSize])
	idx += wSize
	copy(g.Uz.Data, weights.Data[idx:idx+uSize])
	idx += uSize
	copy(g.Ur.Data, weights.Data[idx:idx+uSize])
	idx += uSize
	copy(g.Uh.Data, weights.Data[idx:idx+uSize])

	// Unpack biases
	copy(g.Bz, bias[0:hiddenSize])
	copy(g.Br, bias[hiddenSize:hiddenSize*2])
	copy(g.Bh, bias[hiddenSize*2:hiddenSize*3])
}

// Type returns the layer type name.
func (g *GRULayer) Type() string {
	return "gru"
}

// Helper functions
func sigmoid32(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

func tanh32(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

// randFloat32 returns a random float32 in [0, 1)
// Uses a simple LCG for reproducibility in testing
var randState uint64 = 42

func randFloat32() float32 {
	randState = randState*6364136223846793005 + 1442695040888963407
	return float32(randState>>33) / float32(1<<31)
}

// ResetRand resets the random state for reproducible testing
func ResetRand(seed uint64) {
	randState = seed
}
