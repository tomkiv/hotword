package cmd

import (
	"strings"
	"testing"
)

func TestTrainCommand(t *testing.T) {
	root := NewRootCmd()
	train := NewTrainCmd()
	root.AddCommand(train)

	output, err := executeCommand(root, "train", "--help")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !strings.Contains(output, "hotword train [flags]") {
		t.Errorf("Expected 'hotword train [flags]' in output. Got:\n%s", output)
	}
}

func TestTrainExecution(t *testing.T) {
	tmpDir, cleanup := createDummyData(t)
	defer cleanup()

	root := NewRootCmd()
	train := NewTrainCmd()
	root.AddCommand(train)

	output, err := executeCommand(root, "train", "--data", tmpDir, "--epochs", "1", "--lr", "0.05")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !strings.Contains(output, "Loading dataset from") {
		t.Errorf("Expected training message, got: %s", output)
	}
	if !strings.Contains(output, "Starting training for 1 epochs") {
		t.Errorf("Expected epochs 1, got: %s", output)
	}
}
