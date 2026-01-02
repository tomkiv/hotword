package main

import (
	"fmt"
	"math"
	"github.com/vitalii/hotword/pkg/model"
)

func numericalGradient(f func() float32, p *float32) float32 {
	h := float32(1e-4)
	old := *p
	*p = old + h
	v1 := f()
	*p = old - h
	v2 := f()
	*p = old
	return (v1 - v2) / (2 * h)
}

func main() {
	fmt.Println("Starting Gradient Verification (Numerical Check)...")

	// 1. Dense Layer Check
	input := &model.Tensor{Data: []float32{0.5, -0.2}, Shape: []int{2}}
	weights := &model.Tensor{Data: []float32{0.1, 0.8}, Shape: []int{1, 2}}
	bias := []float32{0.3}
	gradOutput := &model.Tensor{Data: []float32{1.0}, Shape: []int{1}}

	lossFunc := func() float32 {
		out := model.Dense(input, weights, bias)
		return out.Data[0] // Simple linear loss for check
	}

	_, gradWeights, gradBias := model.DenseBackward(input, weights, bias, gradOutput)

	numGradW0 := numericalGradient(lossFunc, &weights.Data[0])
	numGradB0 := numericalGradient(lossFunc, &bias[0])

	fmt.Printf("Dense Weight Grad: Analytical=%.4f, Numerical=%.4f\n", gradWeights.Data[0], numGradW0)
	fmt.Printf("Dense Bias Grad:   Analytical=%.4f, Numerical=%.4f\n", gradBias[0], numGradB0)

	// 2. ReLU Check
	reluIn := &model.Tensor{Data: []float32{0.5, -0.5}, Shape: []int{2}}
	reluGradOut := &model.Tensor{Data: []float32{1.0, 1.0}, Shape: []int{2}}
	gradReLU := model.ReLUBackward(reluIn, reluGradOut)
	
	fmt.Printf("ReLU Grad (pos): Analytical=%.4f, Expected=1.0000\n", gradReLU.Data[0])
	fmt.Printf("ReLU Grad (neg): Analytical=%.4f, Expected=0.0000\n", gradReLU.Data[1])

	// 3. BCE Loss Check
	yPred := []float32{0.6}
	yTrue := []float32{1.0}
	gradBCE := model.BCEGradient(yPred, yTrue)
	
bceLossFunc := func() float32 { return model.BCELoss(yPred, yTrue) }
	numGradBCE := numericalGradient(bceLossFunc, &yPred[0])
	
	fmt.Printf("BCE Grad: Analytical=%.4f, Numerical=%.4f\n", gradBCE[0], numGradBCE)

	fmt.Println("\nVerification complete. Check if Analytical and Numerical values are closely matched.")
}
