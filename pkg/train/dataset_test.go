package train

import (
	"os"
	"path/filepath"
	"testing"
)

func createTestWAV(path string) error {
	// Create a minimal 16-bit PCM WAV file
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Just use the audio package's helper if it was public, but it's not.
	// We'll write a minimal RIFF header manually.
	f.Write([]byte("RIFF"))
	f.Write([]byte{0, 0, 0, 0}) // Size
	f.Write([]byte("WAVE"))
	f.Write([]byte("fmt "))
	f.Write([]byte{16, 0, 0, 0})      // Size
	f.Write([]byte{1, 0, 1, 0})       // PCM, Mono
	f.Write([]byte{0x80, 0x3e, 0, 0}) // 16000 SR
	f.Write([]byte{0, 0, 0, 0})       // ByteRate
	f.Write([]byte{2, 0, 16, 0})      // Align, Bits
	f.Write([]byte("data"))
	f.Write([]byte{0, 0, 0, 0}) // Size

	// Add some samples
	samples := make([]byte, 1000)
	f.Write(samples)

	return nil
}

func TestDataset(t *testing.T) {
	// Create temporary directory structure
	tmpDir, _ := os.MkdirTemp("", "hotword_test")
	defer os.RemoveAll(tmpDir)

	hotwordDir := filepath.Join(tmpDir, "hotword")
	bgDir := filepath.Join(tmpDir, "background")
	os.Mkdir(hotwordDir, 0755)
	os.Mkdir(bgDir, 0755)

	createTestWAV(filepath.Join(hotwordDir, "h1.wav"))
	createTestWAV(filepath.Join(bgDir, "b1.wav"))

	t.Run("Load Dataset", func(t *testing.T) {
		ds, err := LoadDataset(hotwordDir, bgDir)
		if err != nil {
			t.Fatalf("Failed to load dataset: %v", err)
		}

		// Dataset now includes synthetic noise samples (minimum 100)
		// So we expect: 1 hotword + 1 background + 100 synthetic noise = 102 minimum
		if len(ds.Samples) < 102 {
			t.Errorf("Expected at least 102 samples, got %d", len(ds.Samples))
		}

		hotCount := 0
		for _, s := range ds.Samples {
			if s.IsHotword {
				hotCount++
			}
		}
		if hotCount != 1 {
			t.Errorf("Expected 1 hotword sample, got %d", hotCount)
		}
	})
}
