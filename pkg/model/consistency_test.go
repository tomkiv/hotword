package model

import (
	"bytes"
	"fmt"
	"math"
	"testing"
)

func TestModelConsistency(t *testing.T) {
	t.Run("Save Load Inference Match", func(t *testing.T) {
		// 1. Create a model
		weights := NewTensor([]int{1, 10})
		for i := range weights.Data {
			weights.Data[i] = float32(i) * 0.05
		}
		bias := []float32{0.25}

		// 2. Create dummy input
		input := NewTensor([]int{10})
		for i := range input.Data {
			input.Data[i] = 1.0
		}

		// 3. Run initial inference
		originalOutput := Dense(input, weights, bias)
		originalVal := originalOutput.Data[0]

		// 4. Save to buffer
		buf := new(bytes.Buffer)
		if err := SaveModelToWriter(buf, weights, bias); err != nil {
			t.Fatalf("Failed to save: %v", err)
		}

		// 5. Load back
		lWeights, lBias, err := LoadModelFromReader(buf)
		if err != nil {
			t.Fatalf("Failed to load: %v", err)
		}

		// 6. Run inference with loaded model
		loadedOutput := Dense(input, lWeights, lBias)
		loadedVal := loadedOutput.Data[0]

		// 7. Compare results
		if math.Abs(float64(originalVal-loadedVal)) > 1e-7 {
			t.Errorf("Inference result mismatch! Original: %f, Loaded: %f", originalVal, loadedVal)
		}
		
		fmt.Printf("Consistency Check: Original Prob=%.6f, Loaded Prob=%.6f\n", originalVal, loadedVal)
	})
}
