package train

import (
	"fmt"
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

	return ds, nil
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
