package train

import (
	"fmt"

	"github.com/vitalii/hotword/pkg/model"
)

// AugmentorTrainer is an interface for trainers that support dynamic augmentation.
type AugmentorTrainer interface {
	SetAugmentor(a *Augmentor)
	Train(ds *Dataset, epochs int, featureExtractor func([]float32) *model.Tensor)
}

// Trainer manages the training process for a hotword model.
type Trainer struct {
	model        model.Model
	learningRate float32
	augmentor    *Augmentor
}

// NewTrainer creates a new Trainer.
func NewTrainer(m model.Model, lr float32) *Trainer {
	return &Trainer{
		model:        m,
		learningRate: lr,
	}
}

// SetAugmentor sets the augmentor for the trainer.
func (t *Trainer) SetAugmentor(a *Augmentor) {
	t.augmentor = a
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
	// This combined gradient already accounts for the sigmoid derivative,
	// so we skip the sigmoid layer and start from the layer before it.
	grad := &model.Tensor{Data: []float32{prediction - target}, Shape: []int{1}}

	// Determine starting index for backward pass
	// If last layer is sigmoid, skip it (gradient already incorporates sigmoid derivative)
	startIdx := len(layers) - 1
	if startIdx >= 0 && layers[startIdx].Type() == "sigmoid" {
		startIdx--
	}

	for i := startIdx; i >= 0; i-- {
		gradInput, gradWeights, gradBias := layers[i].Backward(inputs[i], grad)

		// If the layer has parameters, update them
		if gradWeights != nil {
			weights, bias := layers[i].Params()
			model.SGDUpdate(weights, gradWeights, t.learningRate)
			model.SGDBiasUpdate(bias, gradBias, t.learningRate)
			// Push updated params back to the layer (essential for GRU/LSTM which return copies)
			layers[i].SetParams(weights, bias)
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
			audioData := sample.Audio
			
			// Apply dynamic augmentation only to hotwords
			if t.augmentor != nil && sample.IsHotword {
				audioData = t.augmentor.Augment(audioData)
			}

			// Convert raw audio to features (e.g. Mel-Spectrogram)
			features := featureExtractor(audioData)

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
