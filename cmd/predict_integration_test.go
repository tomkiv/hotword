package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vitalii/hotword/pkg/model"
)

func TestPredictIntegration(t *testing.T) {
	// 1. Setup
	tmpDir, err := os.MkdirTemp("", "predict_integration")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	wavFile := filepath.Join(tmpDir, "test.wav")
	createDummyWAV(wavFile)

	modelFile := filepath.Join(tmpDir, "model.bin")
	// Create a dummy model (1 Dense layer acting on 2440 features)
	weights := model.NewTensor([]int{1, 2440})
	bias := []float32{0.0}
	m := model.NewSequentialModel(
		model.NewDenseLayer(weights, bias),
		model.NewSigmoidLayer(),
	)
	if err := model.SaveModel(modelFile, m); err != nil {
		t.Fatalf("Failed to save mock model: %v", err)
	}

	// 2. Run predict
	root := NewRootCmd()
	predict := NewPredictCmd()
	root.AddCommand(predict)

	output, err := executeCommand(root, "predict", "--file", wavFile, "--model", modelFile, "--threshold", "0.5")
	if err != nil {
		t.Fatalf("Predict command failed: %v", err)
	}

	// 3. Verify
	if !strings.Contains(output, "Verdict:") {
		t.Errorf("Expected verdict in output, got:\n%s", output)
	}
	if !strings.Contains(output, "Confidence:") {
		t.Errorf("Expected confidence in output, got:\n%s", output)
	}
}
