package train

import (
	"fmt"
	"github.com/vitalii/hotword/pkg/model"
)

// Trainer manages the training process for a hotword model.
type Trainer struct {
	weights      *model.Tensor
	bias         []float32
	learningRate float32
}

// NewTrainer creates a new Trainer.
func NewTrainer(weights *model.Tensor, bias []float32, lr float32) *Trainer {
	return &Trainer{
		weights:      weights,
		bias:         bias,
		learningRate: lr,
	}
}

// TrainStep performs a single training iteration on a single sample.
// Returns the loss before the update.
func (t *Trainer) TrainStep(input *model.Tensor, target float32) float32 {
	// 1. Forward Pass
	output := model.Dense(input, t.weights, t.bias)
	prediction := output.Data[0]

	// 2. Calculate Loss
	loss := model.BCELoss([]float32{prediction}, []float32{target})

	// 3. Backward Pass
	// gradLoss = dL/dy
	gradLoss := model.BCEGradient([]float32{prediction}, []float32{target})
	gradOutput := &model.Tensor{Data: gradLoss, Shape: []int{1}}

	_, gradWeights, gradBias := model.DenseBackward(input, t.weights, t.bias, gradOutput)

	// 4. Update Weights
	model.SGDUpdate(t.weights, gradWeights, t.learningRate)
	model.SGDBiasUpdate(t.bias, gradBias, t.learningRate)

	return loss
}

// Train runs the training loop over the provided dataset for a number of epochs.
func (t *Trainer) Train(ds *Dataset, epochs int, featureExtractor func([]float32) *model.Tensor) {
	for epoch := 1; epoch <= epochs; epoch++ {
		var totalLoss float32
		for _, sample := range ds.Samples {
			// Convert raw audio to features (e.g. Mel-Spectrogram)
			features := featureExtractor(sample.Audio)
			
			target := float32(0.0)
			if sample.IsHotword {
				target = 1.0
			}
			
			loss := t.TrainStep(features, target)
			totalLoss += loss
		}
		fmt.Printf("Epoch %d/%d - Loss: %.4f\n", epoch, epochs, totalLoss/float32(len(ds.Samples)))
	}
}
