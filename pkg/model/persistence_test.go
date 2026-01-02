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
