package cmd

import (
	"strings"
	"testing"
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
	root := NewRootCmd()
	verify := NewVerifyCmd()
	root.AddCommand(verify)

	output, err := executeCommand(root, "verify", "--model", "test.bin", "--data", "test_data")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !strings.Contains(output, "Verifying model test.bin against data in test_data") {
		t.Errorf("Expected verification message, got: %s", output)
	}
}
