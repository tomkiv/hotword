package cmd

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/viper"
)

func TestInitConfig(t *testing.T) {
	tmpDir, _ := os.MkdirTemp("", "hotword_config_test")
	defer os.RemoveAll(tmpDir)

	configPath := filepath.Join(tmpDir, "config.yaml")
	configContent := []byte("train:\n  epochs: 42\n  lr: 0.99\n")
	os.WriteFile(configPath, configContent, 0644)

	// Save original cfgFile and restore after test
	oldCfgFile := cfgFile
	defer func() { cfgFile = oldCfgFile }()

	cfgFile = configPath
	initConfig()

	if viper.GetInt("train.epochs") != 42 {
		t.Errorf("Expected epochs 42, got %d", viper.GetInt("train.epochs"))
	}
	if viper.GetFloat64("train.lr") != 0.99 {
		t.Errorf("Expected lr 0.99, got %f", viper.GetFloat64("train.lr"))
	}
}
