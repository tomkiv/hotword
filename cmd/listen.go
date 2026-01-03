package cmd

import (
	"context"
	"fmt"
	"os"
	"os/exec"
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
var listenModel string
var listenThreshold float32
var listenCooldown int
var listenMinPower float32

// NewListenCmd creates a new listen command
func NewListenCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "listen",
		Short: "Listen for the hotword in real-time",
		Long:  `Listen for the hotword in real-time using the system microphone and trigger an action upon detection.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			modelFile := viper.GetString("listen.model")
			threshold := float32(viper.GetFloat64("listen.threshold"))
			cooldown := viper.GetInt("listen.cooldown")
			minPower := float32(viper.GetFloat64("listen.min_power"))

			if modelFile == "" {
				return fmt.Errorf("model file is required (use --model or set in config)")
			}

			cmd.Printf("Loading model from %s...\n", modelFile)
			weights, bias, err := model.LoadModel(modelFile)
			if err != nil {
				return fmt.Errorf("failed to load model: %w", err)
			}

			// Create model instance
			m := model.NewDenseModel(weights, bias)

			sampleRate := 16000
			e := engine.NewEngine(m, sampleRate)

			device, err := capture.Open("default", sampleRate)
			if err != nil {
				return fmt.Errorf("failed to open audio device: %w", err)
			}
			defer device.Close()

			cmd.Printf("Listening for hotword (Threshold: %.2f, MinPower: %.4f, Cooldown: %dms)...\n", threshold, minPower, cooldown)
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
					// Update VU meter and power level
					_, peak := capture.CalculateLevels(samples)
					bar := capture.GenerateVUBar(peak, 30)
					
					// We MUST update the engine's sliding window even on silence
					// so that it has the correct context when sound starts.
					// However, e.Process currently does both.
					
					// Skip processing during cooldown
					if time.Since(lastDetection) < time.Duration(cooldown)*time.Millisecond {
						fmt.Printf("\rVU: %s [COOLDOWN] Detections: %d\033[K", bar, detectionCount)
						continue
					}

					// Skip inference if audio is too quiet (silence)
					if peak < minPower {
						// We still need to push silence into the engine buffer!
						// Since e.Process does both, we'll just call it but ignore the result
						// or better yet, we'll add a way to just update the buffer.
						e.Process(samples, 2.0) // Threshold 2.0 ensures no detection
						fmt.Printf("\rVU: %s [SILENT] Detections: %d\033[K", bar, detectionCount)
						continue
					}

					confidence, detected := e.Process(samples, threshold)
					fmt.Printf("\rVU: %s Confidence: %.4f | Detections: %d\033[K", bar, confidence, detectionCount)

					if detected {
						detectionCount++
						lastDetection = time.Now()
						fmt.Printf("\n[%s] *** HOTWORD DETECTED! (Confidence: %.4f) ***\n", lastDetection.Format("15:04:05"), confidence)
						
						// Execute actions
						action := viper.GetString("listen.action")
						script := viper.GetString("listen.script")

						if action != "" {
							go executeAction(action)
						}
						if script != "" {
							go executeScript(script)
						}
					}
				}
			}
		},
	}

	cmd.Flags().StringVar(&listenAction, "action", "", "Shell command to execute upon detection")
	cmd.Flags().StringVar(&listenScript, "script", "", "Path to a script to execute upon detection")
	cmd.Flags().StringVar(&listenModel, "model", "model.bin", "Path to the trained model binary")
	cmd.Flags().Float32Var(&listenThreshold, "threshold", 0.5, "Confidence threshold for detection")
	cmd.Flags().IntVar(&listenCooldown, "cooldown", 2000, "Cooldown period in milliseconds after detection")
	cmd.Flags().Float32Var(&listenMinPower, "min-power", 0.001, "Minimum audio power to trigger inference")

	viper.BindPFlag("listen.action", cmd.Flags().Lookup("action"))
	viper.BindPFlag("listen.script", cmd.Flags().Lookup("script"))
	viper.BindPFlag("listen.model", cmd.Flags().Lookup("model"))
	viper.BindPFlag("listen.threshold", cmd.Flags().Lookup("threshold"))
	viper.BindPFlag("listen.cooldown", cmd.Flags().Lookup("cooldown"))
	viper.BindPFlag("listen.min_power", cmd.Flags().Lookup("min-power"))

	return cmd
}

func executeAction(action string) {
	cmd := exec.Command("sh", "-c", action)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Printf("\nAction error: %v\n", err)
	}
}

func executeScript(path string) {
	cmd := exec.Command(path)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Printf("\nScript error: %v\n", err)
	}
}

var listenCmd = NewListenCmd()

func init() {
	rootCmd.AddCommand(listenCmd)
}
