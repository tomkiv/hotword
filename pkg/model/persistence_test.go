package model

import (
	"encoding/binary"
	"os"
	"testing"
)

func TestPersistenceV2(t *testing.T) {
	// Create a complex model
	configs := []LayerConfig{
		{Type: "conv2d", Filters: 4, KernelSize: 3, Stride: 1, Padding: 1},
		{Type: "relu"},
		{Type: "maxpool2d", KernelSize: 2, Stride: 2},
		{Type: "dense", Units: 2},
		{Type: "sigmoid"},
	}
	
	// Input shape: [1, 10, 10]
	mOrig, err := BuildModelFromConfig(configs, []int{1, 10, 10})
	if err != nil {
		t.Fatalf("Failed to build model: %v", err)
	}
	
	tmpFile := "test_model_v2.bin"
	if err := SaveModel(tmpFile, mOrig); err != nil {
		t.Fatalf("SaveModel failed: %v", err)
	}
	defer os.Remove(tmpFile)
	
	mLoaded, err := LoadModel(tmpFile)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	
	layersOrig := mOrig.GetLayers()
	layersLoaded := mLoaded.GetLayers()
	
	if len(layersOrig) != len(layersLoaded) {
		t.Errorf("Expected %d layers, got %d", len(layersOrig), len(layersLoaded))
	}
	
	for i := range layersOrig {
		if layersOrig[i].Type() != layersLoaded[i].Type() {
			t.Errorf("Layer %d type mismatch: expected %s, got %s", i, layersOrig[i].Type(), layersLoaded[i].Type())
		}
	}
	
	// Verify end-to-end inference match
	input := NewTensor([]int{1, 10, 10})
	for i := range input.Data {
		input.Data[i] = 0.5
	}
	
	outOrig := mOrig.Forward(input)
	outLoaded := mLoaded.Forward(input)
	
	if outOrig.Data[0] != outLoaded.Data[0] {
		t.Errorf("Inference mismatch: %f != %f", outOrig.Data[0], outLoaded.Data[0])
	}
}

func TestLoadLegacyV1(t *testing.T) {
	tmpFile := "test_model_v1.bin"
	f, _ := os.Create(tmpFile)
	
	f.Write([]byte("HWMD"))
	binary.Write(f, binary.LittleEndian, uint16(1)) // Version 1
	
	// numRows=1, numCols=2, numBias=1
	binary.Write(f, binary.LittleEndian, uint32(1))
	binary.Write(f, binary.LittleEndian, uint32(2))
	binary.Write(f, binary.LittleEndian, uint32(1))
	
	// weights: [0.1, 0.2]
	binary.Write(f, binary.LittleEndian, float32(0.1))
	binary.Write(f, binary.LittleEndian, float32(0.2))
	
	// bias: [0.5]
	binary.Write(f, binary.LittleEndian, float32(0.5))
	f.Close()
	defer os.Remove(tmpFile)
	
	m, err := LoadModel(tmpFile)
	if err != nil {
		t.Fatalf("LoadModel V1 failed: %v", err)
	}
	
	layers := m.GetLayers()
	if len(layers) != 2 {
		t.Errorf("Expected 2 layers for converted V1 model, got %d", len(layers))
	}
	
	if layers[0].Type() != "dense" || layers[1].Type() != "sigmoid" {
		t.Errorf("Wrong layer types: %s, %s", layers[0].Type(), layers[1].Type())
	}
}
