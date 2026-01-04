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
	ActualLen int // Original length before padding (for masking in variable-length mode)
}

// Dataset contains all loaded training samples.
type Dataset struct {
	Samples []Sample
}

// extractWindows extracts overlapping windows from audio data.
// If audio is shorter than windowLen, returns a single zero-padded window.
// stride determines the overlap between consecutive windows.
func extractWindows(audioData []float32, windowLen, stride int) [][]float32 {
	if len(audioData) == 0 {
		return nil
	}

	// If audio is shorter than window, return single padded window
	if len(audioData) <= windowLen {
		window := make([]float32, windowLen)
		copy(window, audioData)
		return [][]float32{window}
	}

	var windows [][]float32
	for start := 0; start+windowLen <= len(audioData); start += stride {
		window := make([]float32, windowLen)
		copy(window, audioData[start:start+windowLen])
		windows = append(windows, window)
	}

	// Handle remainder: if there's leftover audio, include a final window from the end
	lastStart := len(audioData) - windowLen
	if len(windows) > 0 {
		prevStart := (len(windows) - 1) * stride
		if lastStart > prevStart {
			window := make([]float32, windowLen)
			copy(window, audioData[lastStart:])
			windows = append(windows, window)
		}
	}

	return windows
}

// findOnset detects the start of audio activity using energy-based onset detection.
// Returns the sample index where audio activity begins.
// Uses a sliding window to calculate RMS energy and finds the first window
// that exceeds the threshold relative to the maximum energy in the file.
// leadTime is the number of samples to include before the detected onset.
func findOnset(audioData []float32, sampleRate int, threshold float32, leadTimeSamples int) int {
	if len(audioData) == 0 {
		return 0
	}

	// Window size: 10ms at the given sample rate
	windowSize := sampleRate / 100
	if windowSize < 16 {
		windowSize = 16
	}

	// First pass: find maximum energy to use as reference
	var maxEnergy float32
	for i := 0; i+windowSize <= len(audioData); i += windowSize / 2 {
		energy := calculateRMSEnergy(audioData[i : i+windowSize])
		if energy > maxEnergy {
			maxEnergy = energy
		}
	}

	// If the file is essentially silent, return 0
	if maxEnergy < 0.001 {
		return 0
	}

	// Second pass: find first window that exceeds threshold * maxEnergy
	energyThreshold := threshold * maxEnergy
	for i := 0; i+windowSize <= len(audioData); i += windowSize / 2 {
		energy := calculateRMSEnergy(audioData[i : i+windowSize])
		if energy >= energyThreshold {
			// Found onset, apply lead time (go back a bit before the onset)
			onsetIdx := i - leadTimeSamples
			if onsetIdx < 0 {
				onsetIdx = 0
			}
			return onsetIdx
		}
	}

	// No onset found, return 0
	return 0
}

// calculateRMSEnergy calculates the root mean square energy of a signal window.
func calculateRMSEnergy(samples []float32) float32 {
	if len(samples) == 0 {
		return 0
	}
	var sum float32
	for _, s := range samples {
		sum += s * s
	}
	return float32(sqrt64(float64(sum / float32(len(samples)))))
}

// sqrt64 is a simple square root using Newton's method to avoid math import
func sqrt64(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// cropToOnset extracts a window of targetLen samples starting from the detected onset.
// If there's not enough audio after onset, pads with zeros.
func cropToOnset(audioData []float32, sampleRate, targetLen int, threshold float32) []float32 {
	// Find onset with 50ms lead time
	leadTimeSamples := sampleRate / 20 // 50ms
	onsetIdx := findOnset(audioData, sampleRate, threshold, leadTimeSamples)

	// Extract window starting from onset
	cropped := make([]float32, targetLen)
	available := len(audioData) - onsetIdx
	if available > targetLen {
		available = targetLen
	}
	if available > 0 {
		copy(cropped, audioData[onsetIdx:onsetIdx+available])
	}

	return cropped
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

// LoadDatasetWindowed loads WAV files using overlapping window extraction.
// This extracts multiple 1-second windows from each audio file with the specified stride.
// windowLen is the window size in samples (e.g., 16000 for 1 second at 16kHz).
// stride is the step size between windows (e.g., 8000 for 50% overlap).
func LoadDatasetWindowed(hotwordDir, backgroundDir string, windowLen, stride int) (*Dataset, error) {
	ds := &Dataset{}

	// Load hotwords with windowing
	hotSamples, err := loadFromDirWindowed(hotwordDir, true, windowLen, stride)
	if err != nil {
		return nil, fmt.Errorf("failed to load hotwords: %w", err)
	}
	ds.Samples = append(ds.Samples, hotSamples...)

	// Load background with windowing
	bgSamples, err := loadFromDirWindowed(backgroundDir, false, windowLen, stride)
	if err != nil {
		return nil, fmt.Errorf("failed to load background: %w", err)
	}
	ds.Samples = append(ds.Samples, bgSamples...)

	// Generate synthetic noise samples
	numNoiseSamples := len(hotSamples)
	if numNoiseSamples < 100 {
		numNoiseSamples = 100
	}
	noiseSamples := generateNoiseSamples(numNoiseSamples, windowLen)
	ds.Samples = append(ds.Samples, noiseSamples...)

	return ds, nil
}

// LoadDatasetWithPadding loads WAV files with variable length support.
// Audio shorter than maxLen is padded with zeros at the end.
// Audio longer than maxLen is truncated from the END (preserving the beginning).
// Each sample records its ActualLen for masking during training/inference.
func LoadDatasetWithPadding(hotwordDir, backgroundDir string, maxLen int) (*Dataset, error) {
	ds := &Dataset{}

	// Load hotwords with padding
	hotSamples, err := loadFromDirPadded(hotwordDir, true, maxLen)
	if err != nil {
		return nil, fmt.Errorf("failed to load hotwords: %w", err)
	}
	ds.Samples = append(ds.Samples, hotSamples...)

	// Load background with padding
	bgSamples, err := loadFromDirPadded(backgroundDir, false, maxLen)
	if err != nil {
		return nil, fmt.Errorf("failed to load background: %w", err)
	}
	ds.Samples = append(ds.Samples, bgSamples...)

	// Generate synthetic noise samples
	numNoiseSamples := len(hotSamples)
	if numNoiseSamples < 100 {
		numNoiseSamples = 100
	}
	noiseSamples := generateNoiseSamples(numNoiseSamples, maxLen)
	ds.Samples = append(ds.Samples, noiseSamples...)

	return ds, nil
}

// LoadDatasetWithOnset loads WAV files using onset detection.
// For both hotword and background files: detects where audio activity begins and extracts from there.
// This helps with long files containing silence at the beginning.
// threshold is the onset detection threshold (0.0-1.0), lower = more sensitive.
// sampleRate is needed for onset detection window sizing.
func LoadDatasetWithOnset(hotwordDir, backgroundDir string, targetLen, sampleRate int, threshold float32) (*Dataset, error) {
	ds := &Dataset{}

	// Load hotwords with onset detection
	hotSamples, err := loadFromDirWithOnset(hotwordDir, true, targetLen, sampleRate, threshold)
	if err != nil {
		return nil, fmt.Errorf("failed to load hotwords: %w", err)
	}
	ds.Samples = append(ds.Samples, hotSamples...)

	// Load background with onset detection (finds activity in long silent files)
	bgSamples, err := loadFromDirWithOnset(backgroundDir, false, targetLen, sampleRate, threshold)
	if err != nil {
		return nil, fmt.Errorf("failed to load background: %w", err)
	}
	ds.Samples = append(ds.Samples, bgSamples...)

	// Generate synthetic noise samples
	numNoiseSamples := len(hotSamples)
	if numNoiseSamples < 100 {
		numNoiseSamples = 100
	}
	noiseSamples := generateNoiseSamples(numNoiseSamples, targetLen)
	ds.Samples = append(ds.Samples, noiseSamples...)

	return ds, nil
}

// LoadDatasetWithOnsetAndStride combines onset detection with window extraction.
// For both hotword and background files: finds onset, then extracts multiple overlapping windows.
// This maximizes training data while ensuring all windows contain actual audio activity.
func LoadDatasetWithOnsetAndStride(hotwordDir, backgroundDir string, windowLen, stride, sampleRate int, threshold float32) (*Dataset, error) {
	ds := &Dataset{}

	// Load hotwords with onset detection + windowing
	hotSamples, err := loadFromDirWithOnsetAndStride(hotwordDir, true, windowLen, stride, sampleRate, threshold)
	if err != nil {
		return nil, fmt.Errorf("failed to load hotwords: %w", err)
	}
	ds.Samples = append(ds.Samples, hotSamples...)

	// Load background with onset detection + windowing
	bgSamples, err := loadFromDirWithOnsetAndStride(backgroundDir, false, windowLen, stride, sampleRate, threshold)
	if err != nil {
		return nil, fmt.Errorf("failed to load background: %w", err)
	}
	ds.Samples = append(ds.Samples, bgSamples...)

	// Generate synthetic noise samples
	numNoiseSamples := len(hotSamples)
	if numNoiseSamples < 100 {
		numNoiseSamples = 100
	}
	noiseSamples := generateNoiseSamples(numNoiseSamples, windowLen)
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

			actualLen := len(audioData)
			if actualLen > targetLength {
				actualLen = targetLength
			}

			samples = append(samples, Sample{
				Audio:     normalized,
				IsHotword: isHotword,
				ActualLen: actualLen,
			})
		}
	}
	return samples, nil
}

// loadFromDirWindowed loads audio files and extracts multiple overlapping windows.
func loadFromDirWindowed(dir string, isHotword bool, windowLen, stride int) ([]Sample, error) {
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

			// Extract overlapping windows
			windows := extractWindows(audioData, windowLen, stride)
			for _, window := range windows {
				samples = append(samples, Sample{
					Audio:     window,
					IsHotword: isHotword,
					ActualLen: windowLen, // Windows are always full length
				})
			}
		}
	}
	return samples, nil
}

// loadFromDirPadded loads audio files with variable length support.
// Shorter audio is padded with zeros, longer audio is truncated from the end.
func loadFromDirPadded(dir string, isHotword bool, maxLen int) ([]Sample, error) {
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

			// Determine actual length (before padding/truncation)
			actualLen := len(audioData)
			if actualLen > maxLen {
				actualLen = maxLen
			}

			// Create padded/truncated buffer
			padded := make([]float32, maxLen)
			copy(padded, audioData) // Truncates naturally if audioData > maxLen

			samples = append(samples, Sample{
				Audio:     padded,
				IsHotword: isHotword,
				ActualLen: actualLen, // Records original length for masking
			})
		}
	}
	return samples, nil
}

// loadFromDirWithOnset loads audio files using onset detection.
// Detects where audio activity begins and extracts targetLen samples from that point.
func loadFromDirWithOnset(dir string, isHotword bool, targetLen, sampleRate int, threshold float32) ([]Sample, error) {
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

			// Use onset detection to crop the audio
			cropped := cropToOnset(audioData, sampleRate, targetLen, threshold)

			samples = append(samples, Sample{
				Audio:     cropped,
				IsHotword: isHotword,
				ActualLen: targetLen,
			})
		}
	}
	return samples, nil
}

// loadFromDirWithOnsetAndStride combines onset detection with window extraction.
// First finds onset in each file, then extracts multiple overlapping windows from that point.
func loadFromDirWithOnsetAndStride(dir string, isHotword bool, windowLen, stride, sampleRate int, threshold float32) ([]Sample, error) {
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

			// Find onset with 50ms lead time
			leadTimeSamples := sampleRate / 20 // 50ms
			onsetIdx := findOnset(audioData, sampleRate, threshold, leadTimeSamples)

			// Extract audio from onset to end
			audioFromOnset := audioData[onsetIdx:]

			// Extract overlapping windows from the onset point
			windows := extractWindows(audioFromOnset, windowLen, stride)
			for _, window := range windows {
				samples = append(samples, Sample{
					Audio:     window,
					IsHotword: isHotword,
					ActualLen: windowLen,
				})
			}
		}
	}
	return samples, nil
}
