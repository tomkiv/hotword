package train

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/tomkiv/hotword/pkg/audio"
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

// CropToOnset extracts a window of targetLen samples starting from the detected onset.
// If there's not enough audio after onset, pads with zeros.
// This function is exported for use in predict command.
func CropToOnset(audioData []float32, sampleRate, targetLen int, threshold float32) []float32 {
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

	pb := NewProgressBar(0, "Loading Dataset")

	// Load hotwords
	hotSamples, err := loadFromDir(hotwordDir, true, targetLength, pb)
	if err != nil {
		return nil, fmt.Errorf("failed to load hotwords: %w", err)
	}
	ds.Samples = append(ds.Samples, hotSamples...)

	// Load background
	bgSamples, err := loadFromDir(backgroundDir, false, targetLength, pb)
	if err != nil {
		return nil, fmt.Errorf("failed to load background: %w", err)
	}
	ds.Samples = append(ds.Samples, bgSamples...)

	pb.Finish()

	// Generate synthetic noise samples...
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
	pb := NewProgressBar(0, "Loading Dataset (Windowed)")

	// Load hotwords with windowing
	hotSamples, err := loadFromDirWindowed(hotwordDir, true, windowLen, stride, pb)
	if err != nil {
		return nil, fmt.Errorf("failed to load hotwords: %w", err)
	}
	ds.Samples = append(ds.Samples, hotSamples...)

	// Load background with windowing
	bgSamples, err := loadFromDirWindowed(backgroundDir, false, windowLen, stride, pb)
	if err != nil {
		return nil, fmt.Errorf("failed to load background: %w", err)
	}
	ds.Samples = append(ds.Samples, bgSamples...)

	pb.Finish()

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
func LoadDatasetWithPadding(hotwordDir, backgroundDir string, maxLen int) (*Dataset, error) {
	ds := &Dataset{}
	pb := NewProgressBar(0, "Loading Dataset (Padded)")

	// Load hotwords with padding
	hotSamples, err := loadFromDirPadded(hotwordDir, true, maxLen, pb)
	if err != nil {
		return nil, fmt.Errorf("failed to load hotwords: %w", err)
	}
	ds.Samples = append(ds.Samples, hotSamples...)

	// Load background with padding
	bgSamples, err := loadFromDirPadded(backgroundDir, false, maxLen, pb)
	if err != nil {
		return nil, fmt.Errorf("failed to load background: %w", err)
	}
	ds.Samples = append(ds.Samples, bgSamples...)

	pb.Finish()

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
func LoadDatasetWithOnset(hotwordDir, backgroundDir string, targetLen, sampleRate int, threshold float32) (*Dataset, error) {
	ds := &Dataset{}
	pb := NewProgressBar(0, "Loading Dataset (Onset)")

	// Load hotwords with onset detection
	hotSamples, err := loadFromDirWithOnset(hotwordDir, true, targetLen, sampleRate, threshold, pb)
	if err != nil {
		return nil, fmt.Errorf("failed to load hotwords: %w", err)
	}
	ds.Samples = append(ds.Samples, hotSamples...)

	// Load background with onset detection (finds activity in long silent files)
	bgSamples, err := loadFromDirWithOnset(backgroundDir, false, targetLen, sampleRate, threshold, pb)
	if err != nil {
		return nil, fmt.Errorf("failed to load background: %w", err)
	}
	ds.Samples = append(ds.Samples, bgSamples...)

	pb.Finish()

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
func LoadDatasetWithOnsetAndStride(hotwordDir, backgroundDir string, windowLen, stride, sampleRate int, threshold float32) (*Dataset, error) {
	ds := &Dataset{}
	pb := NewProgressBar(0, "Loading Dataset (Onset+Stride)")

	// Load hotwords with onset detection + windowing
	hotSamples, err := loadFromDirWithOnsetAndStride(hotwordDir, true, windowLen, stride, sampleRate, threshold, pb)
	if err != nil {
		return nil, fmt.Errorf("failed to load hotwords: %w", err)
	}
	ds.Samples = append(ds.Samples, hotSamples...)

	// Load background with onset detection + windowing
	bgSamples, err := loadFromDirWithOnsetAndStride(backgroundDir, false, windowLen, stride, sampleRate, threshold, pb)
	if err != nil {
		return nil, fmt.Errorf("failed to load background: %w", err)
	}
	ds.Samples = append(ds.Samples, bgSamples...)

	pb.Finish()

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

type fileProcessor func(path string) ([]Sample, error)

func parallelLoadFromDir(dir string, isHotword bool, proc fileProcessor, pb *ProgressBar) ([]Sample, error) {
	files, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var wavFiles []string
	for _, f := range files {
		if filepath.Ext(f.Name()) == ".wav" {
			wavFiles = append(wavFiles, filepath.Join(dir, f.Name()))
		}
	}

	if len(wavFiles) == 0 {
		return nil, nil
	}

	numFiles := len(wavFiles)
	if pb != nil {
		// Update total if it was initialized with 0
		if pb.Total == 0 {
			pb.Total = numFiles
		} else {
			// Add to existing total (for hotword + background)
			pb.Total += numFiles
		}
	}

	numWorkers := runtime.NumCPU()
	if numWorkers > numFiles {
		numWorkers = numFiles
	}

	pathsChan := make(chan string, numFiles)
	resultsChan := make(chan []Sample, numFiles)
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range pathsChan {
				samples, err := proc(path)
				if err != nil {
					fmt.Printf("\nError loading %s: %v\n", path, err)
					resultsChan <- nil
					continue
				}
				resultsChan <- samples
			}
		}()
	}

	for _, path := range wavFiles {
		pathsChan <- path
	}
	close(pathsChan)

	// Result collector
	var allSamples []Sample
	done := make(chan bool)
	go func() {
		count := 0
		for samples := range resultsChan {
			if samples != nil {
				allSamples = append(allSamples, samples...)
			}
			count++
			if pb != nil {
				pb.Update(pb.Current + 1)
			}
			if count == numFiles {
				break
			}
		}
		done <- true
	}()

	wg.Wait()
	<-done

	return allSamples, nil
}

func loadFromDir(dir string, isHotword bool, targetLength int, pb *ProgressBar) ([]Sample, error) {
	proc := func(path string) ([]Sample, error) {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer file.Close()

		audioData, _, err := audio.LoadWAV(file)
		if err != nil {
			return nil, err
		}

		// Normalize length
		normalized := make([]float32, targetLength)
		copy(normalized, audioData)

		actualLen := len(audioData)
		if actualLen > targetLength {
			actualLen = targetLength
		}

		return []Sample{{
			Audio:     normalized,
			IsHotword: isHotword,
			ActualLen: actualLen,
		}}, nil
	}
	return parallelLoadFromDir(dir, isHotword, proc, pb)
}

// loadFromDirWindowed loads audio files and extracts multiple overlapping windows.
func loadFromDirWindowed(dir string, isHotword bool, windowLen, stride int, pb *ProgressBar) ([]Sample, error) {
	proc := func(path string) ([]Sample, error) {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer file.Close()

		audioData, _, err := audio.LoadWAV(file)
		if err != nil {
			return nil, err
		}

		windows := extractWindows(audioData, windowLen, stride)
		var samples []Sample
		for _, window := range windows {
			samples = append(samples, Sample{
				Audio:     window,
				IsHotword: isHotword,
				ActualLen: windowLen,
			})
		}
		return samples, nil
	}
	return parallelLoadFromDir(dir, isHotword, proc, pb)
}

// loadFromDirPadded loads audio files with variable length support.
func loadFromDirPadded(dir string, isHotword bool, maxLen int, pb *ProgressBar) ([]Sample, error) {
	proc := func(path string) ([]Sample, error) {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer file.Close()

		audioData, _, err := audio.LoadWAV(file)
		if err != nil {
			return nil, err
		}

		actualLen := len(audioData)
		if actualLen > maxLen {
			actualLen = maxLen
		}

		padded := make([]float32, maxLen)
		copy(padded, audioData)

		return []Sample{{
			Audio:     padded,
			IsHotword: isHotword,
			ActualLen: actualLen,
		}}, nil
	}
	return parallelLoadFromDir(dir, isHotword, proc, pb)
}

// loadFromDirWithOnset loads audio files using onset detection.
func loadFromDirWithOnset(dir string, isHotword bool, targetLen, sampleRate int, threshold float32, pb *ProgressBar) ([]Sample, error) {
	proc := func(path string) ([]Sample, error) {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer file.Close()

		audioData, _, err := audio.LoadWAV(file)
		if err != nil {
			return nil, err
		}

		cropped := CropToOnset(audioData, sampleRate, targetLen, threshold)

		return []Sample{{
			Audio:     cropped,
			IsHotword: isHotword,
			ActualLen: targetLen,
		}}, nil
	}
	return parallelLoadFromDir(dir, isHotword, proc, pb)
}

// loadFromDirWithOnsetAndStride combines onset detection with window extraction.
func loadFromDirWithOnsetAndStride(dir string, isHotword bool, windowLen, stride, sampleRate int, threshold float32, pb *ProgressBar) ([]Sample, error) {
	proc := func(path string) ([]Sample, error) {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer file.Close()

		audioData, _, err := audio.LoadWAV(file)
		if err != nil {
			return nil, err
		}

		leadTimeSamples := sampleRate / 20
		onsetIdx := findOnset(audioData, sampleRate, threshold, leadTimeSamples)
		audioFromOnset := audioData[onsetIdx:]

		windows := extractWindows(audioFromOnset, windowLen, stride)
		var samples []Sample
		for _, window := range windows {
			samples = append(samples, Sample{
				Audio:     window,
				IsHotword: isHotword,
				ActualLen: windowLen,
			})
		}
		return samples, nil
	}
	return parallelLoadFromDir(dir, isHotword, proc, pb)
}
