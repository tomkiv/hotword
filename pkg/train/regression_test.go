package train

import (
	"testing"
	"github.com/tomkiv/hotword/pkg/model"
)

// MockCopyLayer returns copies of its parameters instead of pointers.
type MockCopyLayer struct {
	weights *model.Tensor
	bias    []float32
}

func (l *MockCopyLayer) Forward(input *model.Tensor) *model.Tensor {
	return model.Dense(input, l.weights, l.bias)
}

func (l *MockCopyLayer) ForwardStateful(input *model.Tensor) *model.Tensor {
	return l.Forward(input)
}

func (l *MockCopyLayer) ResetState() {}

func (l *MockCopyLayer) Backward(input, gradOutput *model.Tensor) (*model.Tensor, *model.Tensor, []float32) {
	return model.DenseBackward(input, l.weights, l.bias, gradOutput)
}

func (l *MockCopyLayer) Params() (*model.Tensor, []float32) {
	// Return a COPY
	w := model.NewTensor(l.weights.Shape)
	copy(w.Data, l.weights.Data)
	b := make([]float32, len(l.bias))
	copy(b, l.bias)
	return w, b
}

func (l *MockCopyLayer) SetParams(weights *model.Tensor, bias []float32) {
	copy(l.weights.Data, weights.Data)
	copy(l.bias, bias)
}

func (l *MockCopyLayer) Type() string { return "mock_copy" }

func TestTrainerParameterUpdateRegression(t *testing.T) {
	w := model.NewTensor([]int{1, 2})
	w.Data = []float32{0.5, 0.5}
	b := []float32{0.0}
	
	layer := &MockCopyLayer{weights: w, bias: b}
	m := model.NewSequentialModel(layer)
	trainer := NewTrainer(m, 1.0)
	
	input := model.NewTensor([]int{2})
	input.Data = []float32{1.0, 1.0}
	
	// Initial prediction: 1.0 -> target 0.0 -> gradient exists
	trainer.TrainStep(input, 0.0)
	
	// If the bug exists, w.Data remains [0.5, 0.5] because SGDUpdate modified the copy
	if w.Data[0] == 0.5 {
		t.Error("FAIL: Weights were not updated in the original layer. The trainer likely modified a copy.")
	}
}
