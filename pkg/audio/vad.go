package audio

import (
	"math"
	"time"
)

// VAD implements a lightweight Voice Activity Detector.
type VAD struct {
	EnergyThreshold float32
	ZCRThreshold    float32
	HangoverMs      int
	
	lastSpeechTime  time.Time
}

// NewVAD creates a new VAD instance.
func NewVAD(energyThreshold, zcrThreshold float32, hangoverMs int) *VAD {
	return &VAD{
		EnergyThreshold: energyThreshold,
		ZCRThreshold:    zcrThreshold,
		HangoverMs:      hangoverMs,
	}
}

// IsSpeech returns true if human speech is detected in the provided samples.
// It uses a hybrid approach of RMS energy and Zero-Crossing Rate (ZCR).
func (v *VAD) IsSpeech(samples []float32) bool {
	if len(samples) == 0 {
		return false
	}

	rms := CalculateRMS(samples)
	zcr := CalculateZCR(samples)

	// Speech usually has high energy and low ZCR (low frequency components)
	// Noise (like a fan) often has low energy or high ZCR (high frequency components)
	isCurrentlySpeech := rms >= v.EnergyThreshold && zcr < v.ZCRThreshold

	if isCurrentlySpeech {
		v.lastSpeechTime = time.Now()
		return true
	}

	// Apply hangover logic
	if !v.lastSpeechTime.IsZero() && time.Since(v.lastSpeechTime) < time.Duration(v.HangoverMs)*time.Millisecond {
		return true
	}

	return false
}

// CalculateRMS calculates the Root Mean Square energy of the samples.
func CalculateRMS(samples []float32) float32 {
	if len(samples) == 0 {
		return 0
	}
	var sum float32
	for _, s := range samples {
		sum += s * s
	}
	return float32(math.Sqrt(float64(sum / float32(len(samples)))))
}

// CalculateZCR calculates the Zero-Crossing Rate of the samples.
func CalculateZCR(samples []float32) float32 {
	if len(samples) <= 1 {
		return 0
	}
	count := 0
	for i := 1; i < len(samples); i++ {
		// Zero crossing occurs when sign changes
		if (samples[i-1] > 0 && samples[i] <= 0) || (samples[i-1] < 0 && samples[i] >= 0) {
			count++
		}
	}
	// Return normalized ZCR (0.0 to 1.0)
	return float32(count) / float32(len(samples)-1)
}
