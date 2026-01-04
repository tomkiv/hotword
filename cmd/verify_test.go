package cmd

import (
	"path/filepath"
	"strings"
	"testing"
	"github.com/tomkiv/hotword/pkg/model"
)

func TestVerifyCommand(t *testing.T) {
	root := NewRootCmd()
	verify := NewVerifyCmd()
	root.AddCommand(verify)

	output, err := executeCommand(root, "verify", "--help")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !strings.Contains(output, "hotword verify [flags]") {
		t.Errorf("Expected 'hotword verify [flags]' in output. Got:\n%s", output)
	}
}

func TestVerifyExecution(t *testing.T) {
	tmpDir, cleanup := createDummyData(t)
	defer cleanup()

	// Create a dummy model
	modelFile := filepath.Join(tmpDir, "model.bin")
	weights := model.NewTensor([]int{1, 2440})
	bias := []float32{0.0}
	m := model.NewSequentialModel(
		model.NewDenseLayer(weights, bias),
		model.NewSigmoidLayer(),
	)
	model.SaveModel(modelFile, m)

	root := NewRootCmd()
	verify := NewVerifyCmd()
	root.AddCommand(verify)

	output, err := executeCommand(root, "verify", "--model", modelFile, "--data", tmpDir)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !strings.Contains(output, "Loading model from") {
		t.Errorf("Expected verification message, got: %s", output)
	}
	if !strings.Contains(output, "Accuracy:") {
		t.Errorf("Expected accuracy report, got: %s", output)
	}
}