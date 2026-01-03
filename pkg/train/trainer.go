package train

import (
	"fmt"
	"github.com/vitalii/hotword/pkg/model"
)

// Trainer manages the training process for a hotword model.
type Trainer struct {
	model        model.Model
	learningRate float32
}

// NewTrainer creates a new Trainer.
func NewTrainer(m model.Model, lr float32) *Trainer {
	return &Trainer{
		model:        m,
		learningRate: lr,
	}
}

// TrainStep performs a single training iteration on a single sample.
// Returns the loss before the update.
func (t *Trainer) TrainStep(input *model.Tensor, target float32) float32 {
	// 1. Forward Pass (storing intermediate inputs for backward pass)
	layers := t.model.GetLayers()
	inputs := make([]*model.Tensor, len(layers)+1)
	inputs[0] = input
	
	for i, layer := range layers {
		inputs[i+1] = layer.Forward(inputs[i])
	}
	
	prediction := inputs[len(inputs)-1].Data[0]
	loss := model.BCELoss([]float32{prediction}, []float32{target})

	// 2. Backward Pass
	// dL/dz = prediction - target (numerically stable BCE+Sigmoid gradient)
	grad := &model.Tensor{Data: []float32{prediction - target}, Shape: []int{1}}
	
	for i := len(layers) - 1; i >= 0; i-- {
		gradInput, gradWeights, gradBias := layers[i].Backward(inputs[i], grad)
		
		// If the layer has parameters, update them
		if gradWeights != nil {
			weights, bias := layers[i].Params()
			model.SGDUpdate(weights, gradWeights, t.learningRate)
			model.SGDBiasUpdate(bias, gradBias, t.learningRate)
		}
		
		grad = gradInput
	}

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