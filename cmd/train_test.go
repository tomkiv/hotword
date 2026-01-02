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
	root := NewRootCmd()
	train := NewTrainCmd()
	root.AddCommand(train)

	output, err := executeCommand(root, "train", "--epochs", "5", "--lr", "0.05")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !strings.Contains(output, "Training model with data from data") {
		t.Errorf("Expected training message, got: %s", output)
	}
	if !strings.Contains(output, "Epochs: 5") {
		t.Errorf("Expected epochs 5, got: %s", output)
	}
	if !strings.Contains(output, "LR: 0.050000") {
		t.Errorf("Expected LR 0.05, got: %s", output)
	}
}