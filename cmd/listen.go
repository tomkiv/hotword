package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/vitalii/hotword/pkg/audio/capture"
	"github.com/vitalii/hotword/pkg/engine"
	"github.com/vitalii/hotword/pkg/model"
)

var listenAction string
var listenScript string
var listenThreshold float32
var listenCooldown int

// NewListenCmd creates a new listen command
func NewListenCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "listen",
		Short: "Listen for the hotword in real-time",
		Long:  `Listen for the hotword in real-time using the system microphone and trigger an action upon detection.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			modelFile := viper.GetString("verify.model")
			threshold := float32(viper.GetFloat64("listen.threshold"))
			cooldown := viper.GetInt("listen.cooldown")

			cmd.Printf("Loading model from %s...\n", modelFile)
			weights, bias, err := model.LoadModel(modelFile)
			if err != nil {
				return fmt.Errorf("failed to load model: %w", err)
			}

			sampleRate := 16000
			e := engine.NewEngine(weights, bias, sampleRate)

			device, err := capture.Open("default", sampleRate)
			if err != nil {
				return fmt.Errorf("failed to open audio device: %w", err)
			}
			defer device.Close()

			cmd.Printf("Listening for hotword (Threshold: %.2f, Cooldown: %dms)...\n", threshold, cooldown)
			cmd.Println("Press Ctrl+C to stop.")

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
					fmt.Printf("\nStream error: %v\n", err)
				}
			}()

			var detectionCount int
			var lastDetection time.Time

			for {
				select {
				case <-ctx.Done():
					cmd.Println("\nStopped.")
					return nil
				case samples := <-out:
					// Skip processing during cooldown
					if time.Since(lastDetection) < time.Duration(cooldown)*time.Millisecond {
						continue
					}

					confidence, detected := e.Process(samples, threshold)
					
					// Update VU meter (reusing capture helpers for visual feedback)
					_, peak := capture.CalculateLevels(samples)
					bar := capture.GenerateVUBar(peak, 30)
					fmt.Printf("\rVU: %s Confidence: %.4f | Detections: %d", bar, confidence, detectionCount)

					if detected {
						detectionCount++
						lastDetection = time.Now()
						fmt.Printf("\n[%s] *** HOTWORD DETECTED! (Confidence: %.4f) ***\n", lastDetection.Format("15:04:05"), confidence)
						
						// Phase 3: Action Execution will go here
					}
				}
			}
		},
	}

	cmd.Flags().StringVar(&listenAction, "action", "", "Shell command to execute upon detection")
	cmd.Flags().StringVar(&listenScript, "script", "", "Path to a script to execute upon detection")
	cmd.Flags().Float32Var(&listenThreshold, "threshold", 0.5, "Confidence threshold for detection")
	cmd.Flags().IntVar(&listenCooldown, "cooldown", 2000, "Cooldown period in milliseconds after detection")

	viper.BindPFlag("listen.action", cmd.Flags().Lookup("action"))
	viper.BindPFlag("listen.script", cmd.Flags().Lookup("script"))
	viper.BindPFlag("listen.threshold", cmd.Flags().Lookup("threshold"))
	viper.BindPFlag("listen.cooldown", cmd.Flags().Lookup("cooldown"))

	return cmd
}

var listenCmd = NewListenCmd()

func init() {
	rootCmd.AddCommand(listenCmd)
}
