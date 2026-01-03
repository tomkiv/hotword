package train

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/vitalii/hotword/pkg/audio"
)

// Sample represents a single training sample.
type Sample struct {
	Audio     []float32
	IsHotword bool
}

// Dataset contains all loaded training samples.
type Dataset struct {
	Samples []Sample
}

// LoadDataset loads WAV files from the hotword and background directories.
// It normalizes all samples to a fixed length (16000 samples / 1 second).
// It also generates synthetic noise samples to improve robustness.
func LoadDataset(hotwordDir, backgroundDir string) (*Dataset, error) {
	ds := &Dataset{}

	const targetLength = 16000 // 1 second at 16kHz

	// Load hotwords
	hotSamples, err := loadFromDir(hotwordDir, true, targetLength)
	if err != nil {
		return nil, fmt.Errorf("failed to load hotwords: %w", err)
	}
	ds.Samples = append(ds.Samples, hotSamples...)

	// Load background
	bgSamples, err := loadFromDir(backgroundDir, false, targetLength)
	if err != nil {
		return nil, fmt.Errorf("failed to load background: %w", err)
	}
	ds.Samples = append(ds.Samples, bgSamples...)

	// Generate synthetic noise samples to teach model to reject random noise
	// This is critical because real microphone input contains various noise types
	// Generate as many noise samples as hotwords for balanced training
	numNoiseSamples := len(hotSamples)
	if numNoiseSamples < 100 {
		numNoiseSamples = 100
	}
	noiseSamples := generateNoiseSamples(numNoiseSamples, targetLength)
	ds.Samples = append(ds.Samples, noiseSamples...)

	return ds, nil
}

// generateNoiseSamples creates synthetic noise samples for training robustness.
// Includes: white noise, onset patterns, silence, and low-amplitude random signals.
func generateNoiseSamples(count, length int) []Sample {
	samples := make([]Sample, count)

	for i := 0; i < count; i++ {
		audio := make([]float32, length)

		// Choose noise type - 8 different types
		noiseType := i % 8
		switch noiseType {
		case 0:
			// White noise (full amplitude)
			for j := range audio {
				audio[j] = (rand.Float32()*2 - 1) * 0.5
			}
		case 1:
			// Low amplitude white noise
			for j := range audio {
				audio[j] = (rand.Float32()*2 - 1) * 0.1
			}
		case 2:
			// Near silence with occasional spikes
			for j := range audio {
				if rand.Float32() < 0.01 {
					audio[j] = (rand.Float32()*2 - 1) * 0.3
				} else {
					audio[j] = (rand.Float32()*2 - 1) * 0.01
				}
			}
		case 3:
			// Pure silence
			// audio is already zeroed
		case 4:
			// ONSET PATTERN: First half silence, second half noise
			// This is critical - prevents false positives on audio onset
			for j := length / 2; j < length; j++ {
				audio[j] = (rand.Float32()*2 - 1) * 0.5
			}
		case 5:
			// ONSET PATTERN: Gradual fade in of noise
			for j := range audio {
				fadeIn := float32(j) / float32(length)
				audio[j] = (rand.Float32()*2 - 1) * 0.5 * fadeIn
			}
		case 6:
			// ONSET PATTERN: First 75% silence, last 25% noise
			for j := length * 3 / 4; j < length; j++ {
				audio[j] = (rand.Float32()*2 - 1) * 0.5
			}
		case 7:
			// Very low amplitude continuous noise
			for j := range audio {
				audio[j] = (rand.Float32()*2 - 1) * 0.02
			}
		}

		samples[i] = Sample{
			Audio:     audio,
			IsHotword: false,
		}
	}

	return samples
}

func loadFromDir(dir string, isHotword bool, targetLength int) ([]Sample, error) {
	files, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var samples []Sample
	for _, f := range files {
		if filepath.Ext(f.Name()) == ".wav" {
			path := filepath.Join(dir, f.Name())
			file, err := os.Open(path)
			if err != nil {
				return nil, err
			}

			audioData, _, err := audio.LoadWAV(file)
			file.Close()
			if err != nil {
				continue
			}

			// Normalize length
			normalized := make([]float32, targetLength)
			copy(normalized, audioData)

			samples = append(samples, Sample{
				Audio:     normalized,
				IsHotword: isHotword,
			})
		}
	}
	return samples, nil
}
