package cmd

import (
	"strings"
	"testing"

	"github.com/spf13/viper"
)

func TestListenCommand(t *testing.T) {
	root := NewRootCmd()
	listen := NewListenCmd()
	root.AddCommand(listen)

	output, err := executeCommand(root, "listen", "--help")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !strings.Contains(output, "hotword listen [flags]") {
		t.Errorf("Expected 'hotword listen [flags]' in output. Got:\n%s", output)
	}
}

func TestListenFlags(t *testing.T) {
	root := NewRootCmd()
	listen := NewListenCmd()
	root.AddCommand(listen)

	// Set args to test flag binding
	root.SetArgs([]string{"listen", "--action", "echo hello", "--threshold", "0.8", "--cooldown", "5000"})
	
	// We don't want to actually start the listen loop in a unit test,
	// so we might need to adjust the command to allow short-circuiting or mock the loop.
	// For now, let's just verify the flags are bound to viper correctly.
	
	// Execute without actually running the loop (we'll implement this logic in listen.go)
	// For unit testing flags, we can just look up the values after parsing.
	if err := listen.Flags().Parse([]string{"--action", "echo hello", "--threshold", "0.8", "--cooldown", "5000"}); err != nil {
		t.Fatalf("Failed to parse flags: %v", err)
	}

	if viper.GetString("listen.action") != "echo hello" {
		t.Errorf("Expected viper listen.action 'echo hello', got %s", viper.GetString("listen.action"))
	}
	if viper.GetFloat64("listen.threshold") != 0.8 {
		t.Errorf("Expected viper listen.threshold 0.8, got %f", viper.GetFloat64("listen.threshold"))
	}
	if viper.GetInt("listen.cooldown") != 5000 {
		t.Errorf("Expected viper listen.cooldown 5000, got %d", viper.GetInt("listen.cooldown"))
	}
}
