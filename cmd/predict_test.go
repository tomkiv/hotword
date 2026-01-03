package cmd

import (
	"strings"
	"testing"

	"github.com/spf13/viper"
)

func TestPredictCommand(t *testing.T) {
	root := NewRootCmd()
	predict := NewPredictCmd()
	root.AddCommand(predict)

	output, err := executeCommand(root, "predict", "--help")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !strings.Contains(output, "hotword predict [flags]") {
		t.Errorf("Expected 'hotword predict [flags]' in output. Got:\n%s", output)
	}
}

func TestPredictFlags(t *testing.T) {
	root := NewRootCmd()
	predict := NewPredictCmd()
	root.AddCommand(predict)

	// Parse flags manually to test binding
	if err := predict.Flags().Parse([]string{"--file", "test.wav", "--model", "custom.bin", "--threshold", "0.9"}); err != nil {
		t.Fatalf("Failed to parse flags: %v", err)
	}

	if viper.GetString("predict.file") != "test.wav" {
		t.Errorf("Expected viper predict.file 'test.wav', got %s", viper.GetString("predict.file"))
	}
	if viper.GetString("predict.model") != "custom.bin" {
		t.Errorf("Expected viper predict.model 'custom.bin', got %s", viper.GetString("predict.model"))
	}
	if viper.GetFloat64("predict.threshold") != 0.9 {
		t.Errorf("Expected viper predict.threshold 0.9, got %f", viper.GetFloat64("predict.threshold"))
	}
}
