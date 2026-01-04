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
	"github.com/vitalii/hotword/pkg/audio"
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
var listenDebug bool
var listenVADEnergy float32
var listenVADZCR float32
var listenVADHangover int

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
			debug := viper.GetBool("listen.debug")

			// VAD parameters
			vadEnergy := float32(viper.GetFloat64("listen.vad_energy"))
			vadZCR := float32(viper.GetFloat64("listen.vad_zcr"))
			vadHangover := viper.GetInt("listen.vad_hangover")

			if modelFile == "" {
				return fmt.Errorf("model file is required (use --model or set in config)")
			}

			cmd.Printf("Loading model from %s...\n", modelFile)
			m, err := model.LoadModel(modelFile)
			if err != nil {
				return fmt.Errorf("failed to load model: %w", err)
			}

			sampleRate := 16000
			e := engine.NewEngine(m, sampleRate)
			e.SetVAD(audio.NewVAD(vadEnergy, vadZCR, vadHangover))

			device, err := capture.Open("default", sampleRate)
			if err != nil {
				return fmt.Errorf("failed to open audio device: %w", err)
			}
			defer device.Close()

			cmd.Printf("Listening for hotword (Threshold: %.2f, MinPower: %.4f, Cooldown: %dms)...\n", threshold, minPower, cooldown)
			cmd.Printf("VAD Gate: Energy > %.4f AND ZCR < %.4f (Hangover: %dms)\n", vadEnergy, vadZCR, vadHangover)
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

					// Skip processing during cooldown
					if time.Since(lastDetection) < time.Duration(cooldown)*time.Millisecond {
						fmt.Printf("\rVU: %s [COOLDOWN] Detections: %d\033[K", bar, detectionCount)
						continue
					}

					// Skip inference if audio is too quiet (silence)
					if peak < minPower {
						// Update buffer without running inference or affecting smoothProb
						e.PushSamples(samples)
						if debug {
							fmt.Printf("\n[SILENT] peak=%.4f\n", peak)
						} else {
							fmt.Printf("\rVU: %s [SILENT] Detections: %d\033[K", bar, detectionCount)
						}
						continue
					}

					info := e.ProcessDebug(samples, threshold)
					confidence := info.SmoothProb
					detected := info.Detected

					if debug {
						// Detailed debug output
						fmt.Printf("\n[DEBUG] peak=%.4f raw=%.4f smooth=%.4f consec=%d vad=%v detected=%v\n",
							peak, info.RawProb, info.SmoothProb, info.ConsecutiveHigh, info.VADActive, info.Detected)
					} else {
						status := ""
						if !info.VADActive && peak >= minPower {
							status = "[VAD: INACTIVE]"
						}
						fmt.Printf("\rVU: %s Confidence: %.4f %s | Detections: %d\033[K", bar, confidence, status, detectionCount)
					}

					if detected {
						detectionCount++
						lastDetection = time.Now()
						fmt.Printf("\n[%s] *** HOTWORD DETECTED! (Confidence: %.4f) ***\n", lastDetection.Format("15:04:05"), confidence)

						// Reset engine state to prevent residual probability from affecting next detection
						e.Reset()

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
	cmd.Flags().BoolVar(&listenDebug, "debug", false, "Enable debug output showing raw model probabilities")
	cmd.Flags().Float32Var(&listenVADEnergy, "vad-energy", 0.01, "RMS energy threshold for VAD (speech detection)")
	cmd.Flags().Float32Var(&listenVADZCR, "vad-zcr", 0.5, "Zero-Crossing Rate threshold for VAD (speech detection)")
	cmd.Flags().IntVar(&listenVADHangover, "vad-hangover", 300, "VAD hangover period in milliseconds")

	viper.BindPFlag("listen.action", cmd.Flags().Lookup("action"))
	viper.BindPFlag("listen.script", cmd.Flags().Lookup("script"))
	viper.BindPFlag("listen.model", cmd.Flags().Lookup("model"))
	viper.BindPFlag("listen.threshold", cmd.Flags().Lookup("threshold"))
	viper.BindPFlag("listen.cooldown", cmd.Flags().Lookup("cooldown"))
	viper.BindPFlag("listen.min_power", cmd.Flags().Lookup("min-power"))
	viper.BindPFlag("listen.debug", cmd.Flags().Lookup("debug"))
	viper.BindPFlag("listen.vad_energy", cmd.Flags().Lookup("vad-energy"))
	viper.BindPFlag("listen.vad_zcr", cmd.Flags().Lookup("vad-zcr"))
	viper.BindPFlag("listen.vad_hangover", cmd.Flags().Lookup("vad-hangover"))

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
