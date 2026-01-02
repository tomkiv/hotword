package model

import (
	"bytes"
	"os"
	"testing"
)

func TestSaveModel(t *testing.T) {
	t.Run("Serialize Model", func(t *testing.T) {
		// Input size 4
		// weights [2, 4]
		// bias [2]
		weights := NewTensor([]int{2, 4})
		for i := range weights.Data {
			weights.Data[i] = float32(i)
		}
		bias := []float32{0.5, -0.5}

		buf := new(bytes.Buffer)
		err := SaveModelToWriter(buf, weights, bias)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}

		data := buf.Bytes()
		// Check magic bytes
		if string(data[0:4]) != "HWMD" {
			t.Errorf("Expected magic bytes HWMD, got %s", string(data[0:4]))
		}

		// Expected size:
		// 4 (magic) + 2 (version) + 4 (inputSize) + 4 (numWeights) + 4 (numBias)
		// + 8 * 4 (weights) + 2 * 4 (bias)
		// = 4 + 2 + 4 + 4 + 4 + 32 + 8 = 58 bytes
		expectedSize := 58
		if len(data) != expectedSize {
			t.Errorf("Expected buffer size %d, got %d", expectedSize, len(data))
		}
	})

	t.Run("Save Model to File", func(t *testing.T) {
		weights := NewTensor([]int{1, 1})
		weights.Data[0] = 1.0
		bias := []float32{0.0}

		tmpFile := "/Users/vitalii/.gemini/tmp/4ea7f44ea8ed39b191b36db796e7cfa745b89b5eb30827822379ef6ae57a83d0/model.bin"
		err := SaveModel(tmpFile, weights, bias)
		if err != nil {
			t.Fatalf("Failed to save model: %v", err)
		}
		defer os.Remove(tmpFile)

		// Check if file exists and has content
		info, err := os.Stat(tmpFile)
		if err != nil {
			t.Fatalf("File does not exist: %v", err)
		}
		if info.Size() == 0 {
			t.Error("Saved file is empty")
		}
	})
}

func TestLoadModel(t *testing.T) {
	t.Run("Valid Load", func(t *testing.T) {
		weights := NewTensor([]int{2, 2})
		weights.Data = []float32{1.0, 2.0, 3.0, 4.0}
		bias := []float32{0.5, -0.5}

		buf := new(bytes.Buffer)
		SaveModelToWriter(buf, weights, bias)

		loadedWeights, loadedBias, err := LoadModelFromReader(buf)
		if err != nil {
			t.Fatalf("Failed to load model: %v", err)
		}

		if loadedWeights.Shape[0] != 2 || loadedWeights.Shape[1] != 2 {
			t.Errorf("Expected shape [2, 2], got %v", loadedWeights.Shape)
		}
		if loadedWeights.Data[0] != 1.0 || loadedWeights.Data[3] != 4.0 {
			t.Error("Weights data mismatch")
		}
		if len(loadedBias) != 2 || loadedBias[0] != 0.5 {
			t.Error("Bias data mismatch")
		}
	})

	t.Run("Invalid Magic Bytes", func(t *testing.T) {
		buf := bytes.NewReader([]byte("NOTAMODEL"))
		_, _, err := LoadModelFromReader(buf)
		if err == nil {
			t.Error("Expected error for invalid magic bytes")
		}
	})

	t.Run("Load Model from File", func(t *testing.T) {
		weights := NewTensor([]int{1, 1})
		weights.Data[0] = 99.0
		bias := []float32{1.1}

		tmpFile := "/Users/vitalii/.gemini/tmp/4ea7f44ea8ed39b191b36db796e7cfa745b89b5eb30827822379ef6ae57a83d0/model_load.bin"
		SaveModel(tmpFile, weights, bias)
		defer os.Remove(tmpFile)

		lw, lb, err := LoadModel(tmpFile)
		if err != nil {
			t.Fatalf("Failed to load model: %v", err)
		}

		if lw.Data[0] != 99.0 || lb[0] != 1.1 {
			t.Error("Loaded data mismatch")
		}
	})
}
