package cmd

import (
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var trainDataDir string
var trainModelOut string
var trainEpochs int
var trainLR float32

// NewTrainCmd creates a new train command
func NewTrainCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "train",
		Short: "Train a hotword detection model",
		Long:  `Train a hotword detection model from WAV samples in hotword and background directories.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			cmd.Printf("Training model with data from %s, output to %s\n", trainDataDir, trainModelOut)
			cmd.Printf("Epochs: %d, LR: %f\n", trainEpochs, trainLR)
			return nil
		},
	}

	cmd.Flags().StringVar(&trainDataDir, "data", "data", "Directory containing 'hotword' and 'background' subdirectories")
	cmd.Flags().StringVar(&trainModelOut, "out", "model.bin", "Path to save the trained model")
	cmd.Flags().IntVar(&trainEpochs, "epochs", 10, "Number of training epochs")
	cmd.Flags().Float32Var(&trainLR, "lr", 0.01, "Learning rate")

	viper.BindPFlag("train.data", cmd.Flags().Lookup("data"))
	viper.BindPFlag("train.out", cmd.Flags().Lookup("out"))
	viper.BindPFlag("train.epochs", cmd.Flags().Lookup("epochs"))
	viper.BindPFlag("train.lr", cmd.Flags().Lookup("lr"))

	return cmd
}

var trainCmd = NewTrainCmd()

func init() {
	rootCmd.AddCommand(trainCmd)
}
