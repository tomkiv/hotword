package cmd

import (
	"bytes"
	"testing"

	"github.com/spf13/cobra"
)

func executeCommand(root *cobra.Command, args ...string) (output string, err error) {
	buf := new(bytes.Buffer)
	root.SetOut(buf)
	root.SetErr(buf)
	root.SetArgs(args)

	err = root.Execute()
	return buf.String(), err
}

func TestRootCommand(t *testing.T) {
	root := NewRootCmd()
	output, err := executeCommand(root, "--help")
	
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if output == "" {
		t.Error("Expected help output, got empty string")
	}
}