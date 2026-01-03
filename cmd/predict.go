package cmd

import (
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var predictFile string
var predictModel string
var predictThreshold float32

// NewPredictCmd creates a new predict command
func NewPredictCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "predict",
		Short: "Test a single WAV file against a model",
		Long:  `Test a single WAV file against a trained hotword model and get a verdict and confidence score.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			// This will be implemented in Phase 2
			cmd.Println("Predicting...")
			return nil
		},
	}

	cmd.Flags().StringVar(&predictFile, "file", "", "Path to the WAV file to test")
	cmd.Flags().StringVar(&predictModel, "model", "model.bin", "Path to the trained model binary")
	cmd.Flags().Float32Var(&predictThreshold, "threshold", 0.5, "Confidence threshold for detection")

	viper.BindPFlag("predict.file", cmd.Flags().Lookup("file"))
	viper.BindPFlag("predict.model", cmd.Flags().Lookup("model"))
	viper.BindPFlag("predict.threshold", cmd.Flags().Lookup("threshold"))

	return cmd
}

var predictCmd = NewPredictCmd()

func init() {
	rootCmd.AddCommand(predictCmd)
}
