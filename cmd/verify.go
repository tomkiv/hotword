package cmd

import (
	"fmt"
	"path/filepath"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/vitalii/hotword/pkg/engine"
	"github.com/vitalii/hotword/pkg/model"
	"github.com/vitalii/hotword/pkg/train"
)

var verifyModelFile string
var verifyDataDir string

// NewVerifyCmd creates a new verify command
func NewVerifyCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "verify",
		Short: "Verify a trained hotword model",
		Long:  `Verify a trained hotword model against a labeled dataset of WAV samples.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			modelFile := viper.GetString("verify.model")
			dataDir := viper.GetString("verify.data")
			threshold := float32(0.5) // Default threshold

			cmd.Printf("Loading model from %s...\n", modelFile)
			weights, bias, err := model.LoadModel(modelFile)
			if err != nil {
				return fmt.Errorf("failed to load model: %w", err)
			}

			cmd.Printf("Loading verification dataset from %s...\n", dataDir)
			hotwordDir := filepath.Join(dataDir, "hotword")
			backgroundDir := filepath.Join(dataDir, "background")
			ds, err := train.LoadDataset(hotwordDir, backgroundDir)
			if err != nil {
				return fmt.Errorf("failed to load dataset: %w", err)
			}

			if len(ds.Samples) == 0 {
				return fmt.Errorf("no samples found in dataset")
			}

			e := engine.NewEngine(weights, bias, 16000)

			var tp, tn, fp, fn int
			var failedSamples []string

			cmd.Printf("Verifying %d samples...\n", len(ds.Samples))
			for i, sample := range ds.Samples {
				_, detected := e.Process(sample.Audio, threshold)
				
				if sample.IsHotword {
					if detected {
						tp++
					} else {
						fn++
						failedSamples = append(failedSamples, fmt.Sprintf("Sample %d (Hotword) MISSED", i))
					}
				} else {
					if detected {
						fp++
						failedSamples = append(failedSamples, fmt.Sprintf("Sample %d (Background) TRIGGERED", i))
					} else {
						tn++
					}
				}
			}

			total := len(ds.Samples)
			accuracy := float32(tp+tn) / float32(total) * 100

			cmd.Printf("\nVerification Results:\n")
			cmd.Printf("--------------------\n")
			cmd.Printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, tp+tn, total)
			cmd.Printf("Confusion Matrix:\n")
			cmd.Printf("  TP: %d | FN: %d\n", tp, fn)
			cmd.Printf("  FP: %d | TN: %d\n", fp, tn)

			if len(failedSamples) > 0 {
				cmd.Printf("\nFailed Samples:\n")
				for _, msg := range failedSamples {
					cmd.Printf("  - %s\n", msg)
				}
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&verifyModelFile, "model", "model.bin", "Path to the trained model binary")
	cmd.Flags().StringVar(&verifyDataDir, "data", "data", "Directory containing 'hotword' and 'background' subdirectories")

	viper.BindPFlag("verify.model", cmd.Flags().Lookup("model"))
	viper.BindPFlag("verify.data", cmd.Flags().Lookup("data"))

	return cmd
}
var verifyCmd = NewVerifyCmd()

func init() {
	rootCmd.AddCommand(verifyCmd)
}
