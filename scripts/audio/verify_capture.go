package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/vitalii/hotword/pkg/audio/capture"
)

func main() {
	deviceName := "default"
	if len(os.Args) > 1 {
		deviceName = os.Args[1]
	}

	sampleRate := 16000
	device, err := capture.Open(deviceName, sampleRate)
	if err != nil {
		log.Fatalf("Failed to open device: %v", err)
	}
	defer device.Close()

	fmt.Printf("Capture started on device: %s (%dHz)\n", deviceName, sampleRate)
	fmt.Println("Press Ctrl+C to stop.")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle Ctrl+C
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		cancel()
	}()

	out := make(chan []float32, 10)
	go func() {
		if err := capture.Stream(ctx, device, out); err != nil && err != context.Canceled {
			log.Printf("Stream error: %v", err)
		}
	}()

	for {
		select {
		case <-ctx.Done():
			fmt.Println("\nStopped.")
			return
		case samples := <-out:
			_, peak := capture.CalculateLevels(samples)
			bar := capture.GenerateVUBar(peak, 40)
			fmt.Printf("\rVU: %s Peak: %.4f", bar, peak)
		}
	}
}
