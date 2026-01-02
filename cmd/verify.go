package cmd

import (
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
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
			cmd.Printf("Verifying model %s against data in %s\n", verifyModelFile, verifyDataDir)
			// This will be connected to pkg/model.LoadModel and pkg/engine.Engine in the future
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
