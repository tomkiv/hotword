package main

import (
	"fmt"
	"os"
	"github.com/vitalii/hotword/pkg/audio"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run verify_audio.go <path_to_wav>")
		return
	}

	filePath := os.Args[1]
	f, err := os.Open(filePath)
	if err != nil {
		fmt.Printf("Error opening file: %v\n", err)
		return
	}
	defer f.Close()

	samples, sampleRate, err := audio.LoadWAV(f)
	if err != nil {
		fmt.Printf("Error loading WAV: %v\n", err)
		return
	}

	fmt.Printf("Loaded WAV: %s\n", filePath)
	fmt.Printf("Sample Rate: %d Hz\n", sampleRate)
	fmt.Printf("Samples: %d\n", len(samples))

	// Preprocessing parameters
	windowSize := 512
	hopSize := 256
	numMelFilters := 40

	stft := audio.STFT(samples, windowSize, hopSize)
	melSpec := audio.MelSpectrogram(stft, numMelFilters, windowSize, sampleRate, 0, 8000)

	if len(melSpec) > 0 {
		fmt.Printf("Mel-Spectrogram shape: %d frames x %d mel bins\n", len(melSpec), len(melSpec[0]))
		fmt.Println("Verification Successful!")
	} else {
		fmt.Println("Verification Failed: Mel-Spectrogram is empty.")
	}
}
