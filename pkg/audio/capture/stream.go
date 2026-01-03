package capture

import (
	"context"
)

// Stream capture audio from the device and sends it to the provided channel.
// It stops when the context is cancelled.
func Stream(ctx context.Context, device Device, out chan<- []float32) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			samples, err := device.Read()
			if err != nil {
				return err
			}
			if len(samples) > 0 {
				out <- samples
			}
		}
	}
}
