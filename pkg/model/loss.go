package model

import (
	"math"
)

const epsilon = 1e-7

// BCELoss calculates the Binary Cross-Entropy loss.
func BCELoss(yPred, yTrue []float32) float32 {
	var totalLoss float64
	n := float64(len(yPred))
	
	for i := range yPred {
		pred := float64(yPred[i])
		if pred < epsilon {
			pred = epsilon
		}
		if pred > 1-epsilon {
			pred = 1 - epsilon
		}
		
		trueVal := float64(yTrue[i])
		loss := trueVal*math.Log(pred) + (1-trueVal)*math.Log(1-pred)
		totalLoss += loss
	}
	
	return float32(-totalLoss / n)
}

// BCEGradient calculates the gradient of the Binary Cross-Entropy loss with respect to predictions.
func BCEGradient(yPred, yTrue []float32) []float32 {
	n := float32(len(yPred))
	grad := make([]float32, len(yPred))
	
	for i := range yPred {
		pred := yPred[i]
		if pred < epsilon {
			pred = epsilon
		}
		if pred > 1-epsilon {
			pred = 1 - epsilon
		}
		
		grad[i] = (pred - yTrue[i]) / (pred * (1 - pred)) / n
	}
	
	return grad
}
