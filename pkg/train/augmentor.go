package train

import (
	"math/rand"
	"time"

	"github.com/vitalii/hotword/pkg/audio"
)

// AugmentorConfig defines the parameters for dynamic data augmentation.
type AugmentorConfig struct {
	AugmentProb   float32 `mapstructure:"augment_prob"`
	MaxNoiseRatio float32 `mapstructure:"max_noise_ratio"`
	MaxShiftMs    int     `mapstructure:"max_shift_ms"`
	MaxGainScale  float32 `mapstructure:"max_gain_scale"`
}

// Augmentor handles the on-the-fly augmentation of audio samples.
type Augmentor struct {
	config     AugmentorConfig
	noisePool  []Sample
	rng        *rand.Rand
}

// NewAugmentor creates a new Augmentor with the provided config and noise samples.
func NewAugmentor(config AugmentorConfig, noisePool []Sample) *Augmentor {
	return &Augmentor{
		config:    config,
		noisePool: noisePool,
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// Augment applies a sequence of random transformations to the provided samples.
func (a *Augmentor) Augment(samples []float32) []float32 {
	if a.rng.Float32() > a.config.AugmentProb {
		return samples
	}

	out := make([]float32, len(samples))
	copy(out, samples)

	// 1. Time Shifting
	if a.config.MaxShiftMs > 0 {
		maxShiftSamples := (a.config.MaxShiftMs * 16000) / 1000
		if maxShiftSamples > 0 {
			offset := a.rng.Intn(maxShiftSamples*2) - maxShiftSamples
			out = audio.Shift(out, offset)
		}
	}

	// 2. Volume Scaling
	if a.config.MaxGainScale > 0 {
		gain := 1.0 + (a.rng.Float32()*2-1)*a.config.MaxGainScale
		out = audio.Scale(out, gain)
	}

	// 3. Noise Mixing
	if a.config.MaxNoiseRatio > 0 && len(a.noisePool) > 0 {
		noiseSample := a.noisePool[a.rng.Intn(len(a.noisePool))].Audio
		ratio := a.rng.Float32() * a.config.MaxNoiseRatio
		
		// Pick a random segment from the noise sample if it's longer
		start := 0
		if len(noiseSample) > len(out) {
			start = a.rng.Intn(len(noiseSample) - len(out))
		}
		
		out = audio.MixNoise(out, noiseSample[start:], ratio)
	}

	return out
}
