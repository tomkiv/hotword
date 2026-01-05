package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/tomkiv/hotword/pkg/audio"
	"github.com/tomkiv/hotword/pkg/features"
	"github.com/tomkiv/hotword/pkg/model"
	"github.com/tomkiv/hotword/pkg/train"
)

var predictFile string
var predictModel string
var predictThreshold float32
var predictOnset bool

// NewPredictCmd creates a new predict command
func NewPredictCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "predict",
		Short: "Test a single WAV file against a model",
		Long:  `Test a single WAV file against a trained hotword model and get a verdict and confidence score.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			filePath := viper.GetString("predict.file")
			modelPath := viper.GetString("predict.model")
			threshold := float32(viper.GetFloat64("predict.threshold"))
			onset := viper.GetBool("predict.onset")

			if filePath == "" {
				return fmt.Errorf("WAV file path is required (use --file)")
			}

			// 1. Load Model
			cmd.Printf("Loading model %s...\n", modelPath)
			m, err := model.LoadModel(modelPath)
			if err != nil {
				return fmt.Errorf("failed to load model: %w", err)
			}

			// 2. Load WAV
			f, err := os.Open(filePath)
			if err != nil {
				return fmt.Errorf("failed to open WAV file: %w", err)
			}
			defer f.Close()

			samples, sampleRate, err := audio.LoadWAV(f)
			if err != nil {
				return fmt.Errorf("failed to load WAV data: %w", err)
			}

			// 3. Normalize to 1 second (with optional onset detection)
			const targetLength = 16000
			var normalized []float32

			if onset {
				cmd.Println("Using onset detection...")
				normalized = train.CropToOnset(samples, int(sampleRate), targetLength, 0.1)
			} else {
				normalized = make([]float32, targetLength)
				copy(normalized, samples)
			}

			// 4. Extract Features
			// Note: We use the same parameters as training/verification
			input := features.Extract(normalized, int(sampleRate), 512, 256, 40)
			if input == nil {
				return fmt.Errorf("failed to extract features")
			}

			// 5. Inference
			output := m.Forward(input)
			confidence := output.Data[0]

			// 6. Report
			verdict := "NOT HOTWORD"
			if confidence >= threshold {
				verdict = "HOTWORD"
			}

			duration := float64(len(samples)) / float64(sampleRate)

			cmd.Printf("File: %s\n", filePath)
			cmd.Printf("Metadata: Sample Rate=%dHz, Duration=%.2fs\n", sampleRate, duration)
			if onset {
				cmd.Printf("Preprocessing: Onset detection enabled\n")
			}
			cmd.Printf("--------------------\n")
			cmd.Printf("Confidence: %.4f\n", confidence)
			cmd.Printf("Verdict:    %s (Threshold: %.2f)\n", verdict, threshold)

			return nil
		},
	}

	cmd.Flags().StringVar(&predictFile, "file", "", "Path to the WAV file to test")
	cmd.Flags().StringVar(&predictModel, "model", "model.bin", "Path to the trained model binary")
	cmd.Flags().Float32Var(&predictThreshold, "threshold", 0.5, "Confidence threshold for detection")
	cmd.Flags().BoolVar(&predictOnset, "onset", false, "Use onset detection (match training preprocessing)")

	viper.BindPFlag("predict.file", cmd.Flags().Lookup("file"))
	viper.BindPFlag("predict.model", cmd.Flags().Lookup("model"))
	viper.BindPFlag("predict.threshold", cmd.Flags().Lookup("threshold"))
	viper.BindPFlag("predict.onset", cmd.Flags().Lookup("onset"))

	return cmd
}

var predictCmd = NewPredictCmd()

func init() {
	rootCmd.AddCommand(predictCmd)
}
