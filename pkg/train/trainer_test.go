package train

import (
	"testing"
	"github.com/vitalii/hotword/pkg/model"
)

func TestTrainerStep(t *testing.T) {
	// Simple model: 1 Dense layer (2 inputs -> 1 output)
	weights := model.NewTensor([]int{1, 2})
	weights.Data[0] = 0.5
	weights.Data[1] = 0.5
	bias := []float32{0.0}
	
	learningRate := float32(0.1)
	
	// Single sample: Input [1.0, 1.0], target [1.0] (Hotword)
	sample := Sample{
		Audio:     make([]float32, 100), // Preprocessed features would go here
		IsHotword: true,
	}
	// Let's assume preprocessed features are just [1.0, 1.0] for simplicity in this test
	features := model.NewTensor([]int{2})
	features.Data[0] = 1.0
	features.Data[1] = 1.0
	
	// Initial forward: 1.0*0.5 + 1.0*0.5 + 0.0 = 1.0
	// BCE Loss for pred=1.0, true=1.0 is 0.
	// Let's use a target that causes a gradient. Target = 0.0
	sample.IsHotword = false
	
	m := model.NewSequentialModel(
		model.NewDenseLayer(weights, bias),
		model.NewSigmoidLayer(),
	)
	trainer := NewTrainer(m, learningRate)
	loss := trainer.TrainStep(features, 0.0) // features, target
	
	if loss <= 0 {
		t.Errorf("Expected positive loss for misclassification, got %f", loss)
	}
	
	// Check if weights were updated
	if weights.Data[0] == 0.5 {
		t.Error("Weights were not updated after training step")
	}
}

func TestTrainLoop(t *testing.T) {
	weights := model.NewTensor([]int{1, 2})
	bias := []float32{0.0}
	m := model.NewSequentialModel(
		model.NewDenseLayer(weights, bias),
		model.NewSigmoidLayer(),
	)
	trainer := NewTrainer(m, 0.1)

	ds := &Dataset{
		Samples: []Sample{
			{Audio: []float32{1, 1}, IsHotword: true},
			{Audio: []float32{0, 0}, IsHotword: false},
		},
	}

	extractor := func(audio []float32) *model.Tensor {
		return &model.Tensor{Data: audio, Shape: []int{2}}
	}

	// Just check if it runs without panic
	trainer.Train(ds, 2, extractor)
}
