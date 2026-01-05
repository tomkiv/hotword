package model

import (
	"math"
)

// LSTMLayer implements a Long Short-Term Memory layer.
type LSTMLayer struct {
	// Weights for gates: i (input), f (forget), o (output), g (cell candidate)
	Wi, Wf, Wo, Wg *Tensor
	Ui, Uf, Uo, Ug *Tensor
	Bi, Bf, Bo, Bg []float32

	InputSize  int
	HiddenSize int

	// Stateful inference
	hiddenState []float32
	cellState   []float32

	lastInput          *Tensor
	originalInputShape []int
	hSeq, cSeq         []*Tensor
	iSeq, fSeq, oSeq   []*Tensor
	gSeq               []*Tensor
}

func NewLSTMLayer(inputSize, hiddenSize int) *LSTMLayer {
	scale := float32(math.Sqrt(6.0 / float64(inputSize+hiddenSize)))
	hScale := float32(math.Sqrt(6.0 / float64(2*hiddenSize)))

	initW := func(r, c int, s float32) *Tensor {
		t := NewTensor([]int{r, c})
		for i := range t.Data {
			t.Data[i] = (randFloat32()*2 - 1) * s
		}
		return t
	}

	return &LSTMLayer{
		Wi:         initW(hiddenSize, inputSize, scale),
		Wf:         initW(hiddenSize, inputSize, scale),
		Wo:         initW(hiddenSize, inputSize, scale),
		Wg:         initW(hiddenSize, inputSize, scale),
		Ui:         initW(hiddenSize, hiddenSize, hScale),
		Uf:         initW(hiddenSize, hiddenSize, hScale),
		Uo:         initW(hiddenSize, hiddenSize, hScale),
		Ug:         initW(hiddenSize, hiddenSize, hScale),
		Bi:         make([]float32, hiddenSize),
		Bf:         make([]float32, hiddenSize),
		Bo:         make([]float32, hiddenSize),
		Bg:         make([]float32, hiddenSize),
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
	}
}

func (l *LSTMLayer) Forward(input *Tensor) *Tensor {
	return l.forwardInternal(input, nil, nil)
}

func (l *LSTMLayer) ForwardStateful(input *Tensor) *Tensor {
	if l.hiddenState == nil {
		l.hiddenState = make([]float32, l.HiddenSize)
		l.cellState = make([]float32, l.HiddenSize)
	}
	out := l.forwardInternal(input, l.hiddenState, l.cellState)
	copy(l.hiddenState, out.Data)
	// We need to capture the cell state from the last timestep of the internal forward pass
	copy(l.cellState, l.cSeq[len(l.cSeq)-1].Data)
	return out
}

func (l *LSTMLayer) ResetState() {
	l.hiddenState = nil
	l.cellState = nil
}

func (l *LSTMLayer) forwardInternal(input *Tensor, initialH, initialC []float32) *Tensor {
	var flatInput *Tensor
	var seqLen int

	if len(input.Shape) == 3 {
		l.originalInputShape = input.Shape
		channels, height, width := input.Shape[0], input.Shape[1], input.Shape[2]
		seqLen = height
		featureSize := channels * width
		flatInput = NewTensor([]int{seqLen, featureSize})
		for t := 0; t < seqLen; t++ {
			for c := 0; c < channels; c++ {
				for w := 0; w < width; w++ {
					flatInput.Data[t*featureSize+c*width+w] = input.Data[c*height*width+t*width+w]
				}
			}
		}
	} else {
		l.originalInputShape = nil
		flatInput = input
		seqLen = input.Shape[0]
	}

	l.lastInput = flatInput
	l.hSeq = make([]*Tensor, seqLen+1)
	l.cSeq = make([]*Tensor, seqLen+1)
	l.iSeq = make([]*Tensor, seqLen)
	l.fSeq = make([]*Tensor, seqLen)
	l.oSeq = make([]*Tensor, seqLen)
	l.gSeq = make([]*Tensor, seqLen)

	l.hSeq[0] = NewTensor([]int{l.HiddenSize})
	if initialH != nil {
		copy(l.hSeq[0].Data, initialH)
	}
	l.cSeq[0] = NewTensor([]int{l.HiddenSize})
	if initialC != nil {
		copy(l.cSeq[0].Data, initialC)
	}

	inputDim := flatInput.Shape[1]

	for t := 0; t < seqLen; t++ {
		xt := flatInput.Data[t*inputDim : (t+1)*inputDim]
		h := l.hSeq[t].Data
		c := l.cSeq[t].Data

		iG, fG, oG, gG := make([]float32, l.HiddenSize), make([]float32, l.HiddenSize), make([]float32, l.HiddenSize), make([]float32, l.HiddenSize)

		for j := 0; j < l.HiddenSize; j++ {
			var iS, fS, oS, gS float32
			iS, fS, oS, gS = l.Bi[j], l.Bf[j], l.Bo[j], l.Bg[j]
			for k := 0; k < inputDim; k++ {
				xv := xt[k]
				iS += l.Wi.Data[j*inputDim+k] * xv
				fS += l.Wf.Data[j*inputDim+k] * xv
				oS += l.Wo.Data[j*inputDim+k] * xv
				gS += l.Wg.Data[j*inputDim+k] * xv
			}
			for k := 0; k < l.HiddenSize; k++ {
				hv := h[k]
				iS += l.Ui.Data[j*l.HiddenSize+k] * hv
				fS += l.Uf.Data[j*l.HiddenSize+k] * hv
				oS += l.Uo.Data[j*l.HiddenSize+k] * hv
				gS += l.Ug.Data[j*l.HiddenSize+k] * hv
			}
			iG[j], fG[j], oG[j], gG[j] = sigmoid32(iS), sigmoid32(fS), sigmoid32(oS), tanh32(gS)
		}

		newC, newH := make([]float32, l.HiddenSize), make([]float32, l.HiddenSize)
		for j := 0; j < l.HiddenSize; j++ {
			newC[j] = fG[j]*c[j] + iG[j]*gG[j]
			newH[j] = oG[j] * tanh32(newC[j])
		}

		l.iSeq[t], l.fSeq[t], l.oSeq[t], l.gSeq[t] = &Tensor{Data: iG, Shape: []int{l.HiddenSize}}, &Tensor{Data: fG, Shape: []int{l.HiddenSize}}, &Tensor{Data: oG, Shape: []int{l.HiddenSize}}, &Tensor{Data: gG, Shape: []int{l.HiddenSize}}
		l.cSeq[t+1], l.hSeq[t+1] = &Tensor{Data: newC, Shape: []int{l.HiddenSize}}, &Tensor{Data: newH, Shape: []int{l.HiddenSize}}
	}
	return l.hSeq[seqLen]
}

func (l *LSTMLayer) Backward(input, gradOutput *Tensor) (*Tensor, *Tensor, []float32) {
	seqLen := len(l.iSeq)
	inputDim := l.lastInput.Shape[1]
	dWi, dWf, dWo, dWg := NewTensor(l.Wi.Shape), NewTensor(l.Wf.Shape), NewTensor(l.Wo.Shape), NewTensor(l.Wg.Shape)
	dUi, dUf, dUo, dUg := NewTensor(l.Ui.Shape), NewTensor(l.Uf.Shape), NewTensor(l.Uo.Shape), NewTensor(l.Ug.Shape)
	dBi, dBf, dBo, dBg := make([]float32, l.HiddenSize), make([]float32, l.HiddenSize), make([]float32, l.HiddenSize), make([]float32, l.HiddenSize)

	dh := make([]float32, l.HiddenSize)
	copy(dh, gradOutput.Data)
	dc := make([]float32, l.HiddenSize)
	dInput := NewTensor(l.lastInput.Shape)

	for t := seqLen - 1; t >= 0; t-- {
		xt := l.lastInput.Data[t*inputDim : (t+1)*inputDim]
		hPrev := l.hSeq[t].Data
		cPrev := l.cSeq[t].Data
		cCurr := l.cSeq[t+1].Data
		i, f, o, g := l.iSeq[t].Data, l.fSeq[t].Data, l.oSeq[t].Data, l.gSeq[t].Data

		// Store per-timestep gate deltas for BPTT
		diArr := make([]float32, l.HiddenSize)
		dfArr := make([]float32, l.HiddenSize)
		doArr := make([]float32, l.HiddenSize)
		dgArr := make([]float32, l.HiddenSize)
		dcCurrArr := make([]float32, l.HiddenSize)

		for j := 0; j < l.HiddenSize; j++ {
			tc := tanh32(cCurr[j])
			doVal := dh[j] * tc * o[j] * (1 - o[j])
			dcCurr := dc[j] + dh[j]*o[j]*(1-tc*tc)
			dfVal := dcCurr * cPrev[j] * f[j] * (1 - f[j])
			diVal := dcCurr * g[j] * i[j] * (1 - i[j])
			dgVal := dcCurr * i[j] * (1 - g[j]*g[j])

			diArr[j], dfArr[j], doArr[j], dgArr[j] = diVal, dfVal, doVal, dgVal
			dcCurrArr[j] = dcCurr

			dBi[j], dBf[j], dBo[j], dBg[j] = dBi[j]+diVal, dBf[j]+dfVal, dBo[j]+doVal, dBg[j]+dgVal
			for k := 0; k < inputDim; k++ {
				xv := xt[k]
				dWi.Data[j*inputDim+k] += diVal * xv
				dWf.Data[j*inputDim+k] += dfVal * xv
				dWo.Data[j*inputDim+k] += doVal * xv
				dWg.Data[j*inputDim+k] += dgVal * xv
				dInput.Data[t*inputDim+k] += diVal*l.Wi.Data[j*inputDim+k] + dfVal*l.Wf.Data[j*inputDim+k] + doVal*l.Wo.Data[j*inputDim+k] + dgVal*l.Wg.Data[j*inputDim+k]
			}
			for k := 0; k < l.HiddenSize; k++ {
				hv := hPrev[k]
				dUi.Data[j*l.HiddenSize+k] += diVal * hv
				dUf.Data[j*l.HiddenSize+k] += dfVal * hv
				dUo.Data[j*l.HiddenSize+k] += doVal * hv
				dUg.Data[j*l.HiddenSize+k] += dgVal * hv
			}
		}

		// Use per-timestep gate deltas (not accumulated bias gradients) for BPTT
		newDh, newDc := make([]float32, l.HiddenSize), make([]float32, l.HiddenSize)
		for k := 0; k < l.HiddenSize; k++ {
			for j := 0; j < l.HiddenSize; j++ {
				newDh[k] += diArr[j]*l.Ui.Data[j*l.HiddenSize+k] + dfArr[j]*l.Uf.Data[j*l.HiddenSize+k] + doArr[j]*l.Uo.Data[j*l.HiddenSize+k] + dgArr[j]*l.Ug.Data[j*l.HiddenSize+k]
			}
			newDc[k] = dcCurrArr[k] * f[k]
		}
		dh, dc = newDh, newDc
	}

	combinedGradWeights := NewTensor([]int{len(dWi.Data)*4 + len(dUi.Data)*4})
	idx := 0
	for _, tw := range []*Tensor{dWi, dWf, dWo, dWg, dUi, dUf, dUo, dUg} {
		copy(combinedGradWeights.Data[idx:], tw.Data)
		idx += len(tw.Data)
	}
	combinedGradBias := make([]float32, l.HiddenSize*4)
	for i, tb := range [][]float32{dBi, dBf, dBo, dBg} {
		copy(combinedGradBias[i*l.HiddenSize:], tb)
	}

	if l.originalInputShape != nil {
		finalGradInput := NewTensor(l.originalInputShape)
		c, h, w := l.originalInputShape[0], l.originalInputShape[1], l.originalInputShape[2]
		for t := 0; t < h; t++ {
			for ci := 0; ci < c; ci++ { // Bug in var name fixed here
				for wi := 0; wi < w; wi++ {
					finalGradInput.Data[ci*h*w+t*w+wi] = dInput.Data[t*(c*w)+ci*w+wi]
				}
			}
		}
		return finalGradInput, combinedGradWeights, combinedGradBias
	}
	return dInput, combinedGradWeights, combinedGradBias
}

func (l *LSTMLayer) Params() (*Tensor, []float32) {
	totalW := (l.HiddenSize * l.InputSize * 4) + (l.HiddenSize * l.HiddenSize * 4)
	w := NewTensor([]int{totalW})
	idx := 0
	for _, tw := range []*Tensor{l.Wi, l.Wf, l.Wo, l.Wg, l.Ui, l.Uf, l.Uo, l.Ug} {
		copy(w.Data[idx:], tw.Data)
		idx += len(tw.Data)
	}
	b := make([]float32, l.HiddenSize*4)
	for i, tb := range [][]float32{l.Bi, l.Bf, l.Bo, l.Bg} {
		copy(b[i*l.HiddenSize:], tb)
	}
	return w, b
}

func (l *LSTMLayer) SetParams(weights *Tensor, bias []float32) {
	inputSize, hiddenSize := l.InputSize, l.HiddenSize
	wSize, uSize := hiddenSize*inputSize, hiddenSize*hiddenSize
	idx := 0
	for _, tw := range []*Tensor{l.Wi, l.Wf, l.Wo, l.Wg} {
		copy(tw.Data, weights.Data[idx:idx+wSize])
		idx += wSize
	}
	for _, tu := range []*Tensor{l.Ui, l.Uf, l.Uo, l.Ug} {
		copy(tu.Data, weights.Data[idx:idx+uSize])
		idx += uSize
	}
	for i, tb := range [][]float32{l.Bi, l.Bf, l.Bo, l.Bg} {
		copy(tb, bias[i*hiddenSize:(i+1)*hiddenSize])
	}
}

func (l *LSTMLayer) Type() string { return "lstm" }
