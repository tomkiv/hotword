package capture

import (
	"context"
	"testing"
	"time"
)

type mockDevice struct {
	samples []float32
}

func (m *mockDevice) Read() ([]float32, error) {
	return m.samples, nil
}

func (m *mockDevice) Close() error {
	return nil
}

func TestStream(t *testing.T) {
	mock := &mockDevice{samples: []float32{0.1, 0.2, 0.3}}
	out := make(chan []float32, 1)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	go func() {
		Stream(ctx, mock, out)
	}()

	select {
	case samples := <-out:
		if len(samples) != 3 {
			t.Errorf("Expected 3 samples, got %d", len(samples))
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("Timed out waiting for samples")
	}
}
