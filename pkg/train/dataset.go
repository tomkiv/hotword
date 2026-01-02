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
func LoadDataset(hotwordDir, backgroundDir string) (*Dataset, error) {
	ds := &Dataset{}

	// Load hotwords
	hotSamples, err := loadFromDir(hotwordDir, true)
	if err != nil {
		return nil, fmt.Errorf("failed to load hotwords: %w", err)
	}
	ds.Samples = append(ds.Samples, hotSamples...)

	// Load background
	bgSamples, err := loadFromDir(backgroundDir, false)
	if err != nil {
		return nil, fmt.Errorf("failed to load background: %w", err)
	}
	ds.Samples = append(ds.Samples, bgSamples...)

	return ds, nil
}

func loadFromDir(dir string, isHotword bool) ([]Sample, error) {
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
				// Log warning and skip?
				continue
			}

			samples = append(samples, Sample{
				Audio:     audioData,
				IsHotword: isHotword,
			})
		}
	}
	return samples, nil
}
