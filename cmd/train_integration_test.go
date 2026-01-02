package cmd

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTrainIntegration(t *testing.T) {
	// 1. Setup temporary directories and dummy WAVs
	tmpDir, err := os.MkdirTemp("", "train_integration")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	hotwordDir := filepath.Join(tmpDir, "hotword")
	bgDir := filepath.Join(tmpDir, "background")
	os.MkdirAll(hotwordDir, 0755)
	os.MkdirAll(bgDir, 0755)

	// Create dummy WAVs
	createDummyWAV(filepath.Join(hotwordDir, "sample.wav"))
	createDummyWAV(filepath.Join(bgDir, "bg.wav"))

	modelOut := filepath.Join(tmpDir, "test_model.bin")

	// 2. Run the train command
	root := NewRootCmd()
	train := NewTrainCmd()
	root.AddCommand(train)

	root.SetArgs([]string{"train", "--data", tmpDir, "--out", modelOut, "--epochs", "1"})
	if err := root.Execute(); err != nil {
		t.Fatalf("Train command failed: %v", err)
	}

	// 3. Verify model file exists
	if _, err := os.Stat(modelOut); os.IsNotExist(err) {
		t.Error("Expected model file to be created, but it doesn't exist")
	}
}
