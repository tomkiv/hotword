package cmd

import (
	"testing"

	"github.com/spf13/viper"
)

func TestViperBinding(t *testing.T) {
	root := NewRootCmd()
	train := NewTrainCmd()
	root.AddCommand(train)

	// Set args to override defaults
	root.SetArgs([]string{"train", "--epochs", "99", "--lr", "0.123"})
	if err := root.Execute(); err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	// Verify Viper values
	if viper.GetInt("train.epochs") != 99 {
		t.Errorf("Expected viper train.epochs 99, got %d", viper.GetInt("train.epochs"))
	}
	if viper.GetFloat64("train.lr") != 0.123 {
		t.Errorf("Expected viper train.lr 0.123, got %f", viper.GetFloat64("train.lr"))
	}
}
