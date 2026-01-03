//go:build !linux && !darwin

package capture

import (
	"errors"
)

// Open opens an audio device for capture.
func Open(deviceName string, sampleRate int) (Device, error) {
	return nil, errors.New("ALSA capture is only supported on Linux")
}
