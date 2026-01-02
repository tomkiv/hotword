package model

import (
	"math"
	"testing"
)

func TestBCELoss(t *testing.T) {
	t.Run("Loss Calculation", func(t *testing.T) {
		yPred := []float32{0.9, 0.1, 0.8}
		yTrue := []float32{1.0, 0.0, 1.0}
		
		loss := BCELoss(yPred, yTrue)
		
		// BCE = - ( (1 * log(0.9) + (1-1) * log(1-0.9)) + 
		//           (0 * log(0.1) + (1-0) * log(1-0.1)) + 
		//           (1 * log(0.8) + (1-1) * log(1-0.8)) ) / 3
		// BCE = - (log(0.9) + log(0.9) + log(0.8)) / 3
		// BCE = - (-0.10536 - 0.10536 - 0.22314) / 3 = 0.43386 / 3 = 0.14462
		expected := float32(-(math.Log(0.9) + math.Log(0.9) + math.Log(0.8)) / 3.0)
		
		if math.Abs(float64(loss-expected)) > 1e-6 {
			t.Errorf("Expected loss %f, got %f", expected, loss)
		}
	})

	t.Run("Gradient Calculation", func(t *testing.T) {
		yPred := []float32{0.5, 0.1, 0.8}
		yTrue := []float32{1.0, 0.0, 1.0}
		
		grad := BCEGradient(yPred, yTrue)
		
		// dL/dy = (yPred - yTrue) / (yPred * (1 - yPred)) / N
		// grad[0] = (0.5 - 1.0) / (0.5 * 0.5) / 3 = -0.5 / 0.25 / 3 = -2 / 3 = -0.666667
		// grad[1] = (0.1 - 0.0) / (0.1 * 0.9) / 3 = 0.1 / 0.09 / 3 = 1.111111 / 3 = 0.370370
		// grad[2] = (0.8 - 1.0) / (0.8 * 0.2) / 3 = -0.2 / 0.16 / 3 = -1.25 / 3 = -0.416667
		
		expected := []float32{
			float32((0.5 - 1.0) / (0.5 * 0.5) / 3.0),
			float32((0.1 - 0.0) / (0.1 * 0.9) / 3.0),
			float32((0.8 - 1.0) / (0.8 * 0.2) / 3.0),
		}
		
		for i := range grad {
			if math.Abs(float64(grad[i]-expected[i])) > 1e-6 {
				t.Errorf("At index %d: expected gradient %f, got %f", i, expected[i], grad[i])
			}
		}
	})
}
