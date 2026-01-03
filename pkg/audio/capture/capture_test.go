package capture

import (
	"testing"
)

func TestOpenDevice(t *testing.T) {
	// For now, we expect this to fail or return an error if no device is found
	// or if we are on a non-Linux platform without ALSA.
	device, err := Open("default", 16000)
	if err != nil {
		t.Logf("Expected error on non-Linux platform: %v", err)
		return
	}
	defer device.Close()
	
	if device == nil {
		t.Error("Expected device object, got nil")
	}
}
