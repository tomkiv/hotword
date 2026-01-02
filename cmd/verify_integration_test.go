package cmd

import (
	"os"
	"path/filepath"
	"testing"
	"github.com/vitalii/hotword/pkg/model"
)

func TestVerifyIntegration(t *testing.T) {
	// 1. Setup temporary directory, dummy WAVs, and a mock model
	tmpDir, err := os.MkdirTemp("", "verify_integration")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	hotwordDir := filepath.Join(tmpDir, "hotword")
	bgDir := filepath.Join(tmpDir, "background")
	os.MkdirAll(hotwordDir, 0755)
	os.MkdirAll(bgDir, 0755)

	createDummyWAV(filepath.Join(hotwordDir, "sample.wav"))
	createDummyWAV(filepath.Join(bgDir, "bg.wav"))

	modelFile := filepath.Join(tmpDir, "test_model.bin")
	
	// Create a dummy model binary
	// Input size for 1s audio: 2440
	weights := model.NewTensor([]int{1, 2440})
	bias := []float32{0.0}
	if err := model.SaveModel(modelFile, weights, bias); err != nil {
		t.Fatalf("Failed to save mock model: %v", err)
	}

	// 2. Run the verify command
	root := NewRootCmd()
	verify := NewVerifyCmd()
	root.AddCommand(verify)

	root.SetArgs([]string{"verify", "--model", modelFile, "--data", tmpDir})
	if err := root.Execute(); err != nil {
		t.Fatalf("Verify command failed: %v", err)
	}

	// 3. (Optional) Check output for "Accuracy" or similar
}
