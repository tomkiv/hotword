package capture

import (
	"math"
	"strings"
)

// CalculateLevels returns the RMS and peak level of the provided audio samples.
func CalculateLevels(samples []float32) (float32, float32) {
	if len(samples) == 0 {
		return 0, 0
	}

	var sumSq float64
	var peak float32

	for _, s := range samples {
		abs := float32(math.Abs(float64(s)))
		if abs > peak {
			peak = abs
		}
		sumSq += float64(s * s)
	}

	rms := float32(math.Sqrt(sumSq / float64(len(samples))))
	return rms, peak
}

// GenerateVUBar returns an ASCII bar representation of the provided level (0.0 to 1.0).
func GenerateVUBar(level float32, width int) string {
	if level < 0 {
		level = 0
	}
	if level > 1 {
		level = 1
	}

	filled := int(math.Round(float64(level * float32(width))))
	bar := strings.Repeat("#", filled) + strings.Repeat(" ", width-filled)
	return "[" + bar + "]"
}
