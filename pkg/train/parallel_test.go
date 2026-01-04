package train

import (
	"testing"

	"github.com/tomkiv/hotword/pkg/model"
)

func TestParallelTrainer(t *testing.T) {
	// Simple model
	w := model.NewTensor([]int{1, 10})
	b := []float32{0.0}
	m := model.NewSequentialModel(model.NewDenseLayer(w, b), model.NewSigmoidLayer())
	
	trainer := NewParallelTrainer(m, 0.1, 2)
	
	// Create dataset
	samples := []Sample{
		{Audio: make([]float32, 10), IsHotword: true},
		{Audio: make([]float32, 10), IsHotword: false},
		{Audio: make([]float32, 10), IsHotword: true},
		{Audio: make([]float32, 10), IsHotword: false},
	}
	ds := &Dataset{Samples: samples}
	
	extractor := func(s []float32) *model.Tensor {
		return &model.Tensor{Data: s, Shape: []int{10}}
	}
	
	// Should run without error
	trainer.Train(ds, 2, extractor)
}
