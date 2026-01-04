package main

import (
	"fmt"
	"os"
	"path/filepath"
	"github.com/tomkiv/hotword/pkg/train"
)

func main() {
	fmt.Println("Starting Dataset Verification...")

	// Create dummy directories if they don't exist
	baseDir := "data_test"
	hotwordDir := filepath.Join(baseDir, "hotword")
	bgDir := filepath.Join(baseDir, "background")

	_ = os.MkdirAll(hotwordDir, 0755)
	_ = os.MkdirAll(bgDir, 0755)

	fmt.Printf("Crawling directories:\n  Hotword: %s\n  Background: %s\n", hotwordDir, bgDir)

	ds, err := train.LoadDataset(hotwordDir, bgDir)
	if err != nil {
		fmt.Printf("Error loading dataset: %v\n", err)
		return
	}

	hotCount := 0
	bgCount := 0
	for _, s := range ds.Samples {
		if s.IsHotword {
			hotCount++
		} else {
			bgCount++
		}
	}

	fmt.Printf("Found %d hotword samples.\n", hotCount)
	fmt.Printf("Found %d background samples.\n", bgCount)
	
	if hotCount == 0 && bgCount == 0 {
		fmt.Println("Note: No .wav files were found in the directories. Add some to test real loading.")
	} else {
		fmt.Println("Dataset loaded successfully!")
	}
}
