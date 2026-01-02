package model

// SGDUpdate performs a single step of Stochastic Gradient Descent.
// It updates the parameters in place: param = param - learningRate * gradient
func SGDUpdate(param, gradient *Tensor, learningRate float32) {
	for i := range param.Data {
		param.Data[i] -= learningRate * gradient.Data[i]
	}
}

// SGDBiasUpdate performs SGD update for bias slices.
func SGDBiasUpdate(bias, gradient []float32, learningRate float32) {
	for i := range bias {
		bias[i] -= learningRate * gradient[i]
	}
}
