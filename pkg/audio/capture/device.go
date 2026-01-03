package capture

import (
	"errors"
)

// Device represents an audio capture device.
type Device interface {
	Read() ([]float32, error)
	Close() error
}

// ErrDeviceClosed is returned when an operation is performed on a closed device.
var ErrDeviceClosed = errors.New("device is closed")
