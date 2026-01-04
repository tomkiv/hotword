package model

import (
	"os"
	"testing"
)

func TestLSTMPersistence(t *testing.T) {
	configs := []LayerConfig{
		{Type: "lstm", Units: 4},
		{Type: "dense", Units: 1},
		{Type: "sigmoid"},
	}
	
	inputShape := []int{1, 10, 10}
	mOrig, _ := BuildModelFromConfig(configs, inputShape)
	
	tmpFile := "lstm_test.bin"
	SaveModel(tmpFile, mOrig)
	defer os.Remove(tmpFile)
	
	mLoaded, err := LoadModel(tmpFile)
	if err != nil {
		t.Fatalf("Failed to load LSTM model: %v", err)
	}
	
	if len(mLoaded.GetLayers()) != 3 {
		t.Errorf("Expected 3 layers, got %d", len(mLoaded.GetLayers()))
	}
	
	if mLoaded.GetLayers()[0].Type() != "lstm" {
		t.Errorf("Expected first layer to be lstm, got %s", mLoaded.GetLayers()[0].Type())
	}
	
	input := NewTensor(inputShape)
	for i := range input.Data {
		input.Data[i] = 0.5
	}
	
	out1 := mOrig.Forward(input)
	out2 := mLoaded.Forward(input)
	
	if out1.Data[0] != out2.Data[0] {
		t.Errorf("Persistence mismatch: %f != %f", out1.Data[0], out2.Data[0])
	}
}
