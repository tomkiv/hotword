package cmd

import (
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
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
			// This will be implemented in Phase 2
			cmd.Println("Listening for hotword...")
			return nil
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
