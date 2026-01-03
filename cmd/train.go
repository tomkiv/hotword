package cmd

import (
	"fmt"
	"math/rand"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/vitalii/hotword/pkg/features"
	"github.com/vitalii/hotword/pkg/model"
	"github.com/vitalii/hotword/pkg/train"
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
			data := viper.GetString("train.data")
			out := viper.GetString("train.out")
			epochs := viper.GetInt("train.epochs")
			lr := float32(viper.GetFloat64("train.lr"))

			sampleRate := 16000
			windowSize := 512
			hopSize := 256
			numMelFilters := 40

			cmd.Printf("Loading dataset from %s...\n", data)
			hotwordDir := filepath.Join(data, "hotword")
			backgroundDir := filepath.Join(data, "background")
			
			ds, err := train.LoadDataset(hotwordDir, backgroundDir)
			if err != nil {
				return fmt.Errorf("failed to load dataset: %w", err)
			}

			if len(ds.Samples) == 0 {
				return fmt.Errorf("no samples found in dataset")
			}

			// Shuffle dataset
			rand.Seed(time.Now().UnixNano())
			rand.Shuffle(len(ds.Samples), func(i, j int) {
				ds.Samples[i], ds.Samples[j] = ds.Samples[j], ds.Samples[i]
			})

			// Define feature extractor
			extractor := func(samples []float32) *model.Tensor {
				return features.Extract(samples, sampleRate, windowSize, hopSize, numMelFilters)
			}

			// Get model configuration from Viper
			var modelConfigs []model.LayerConfig
			if err := viper.UnmarshalKey("model.layers", &modelConfigs); err != nil {
				return fmt.Errorf("failed to parse model configuration: %w", err)
			}

			// If no model config, provide a default single-layer model for backward compatibility
			if len(modelConfigs) == 0 {
				modelConfigs = []model.LayerConfig{
					{Type: "dense", Units: 1},
					{Type: "sigmoid"},
				}
			}

			// Determine input shape from first sample
			firstFeatures := extractor(ds.Samples[0].Audio)
			inputShape := firstFeatures.Shape

			// Build model with Xavier initialization
			m, err := model.BuildModelFromConfig(modelConfigs, inputShape)
			if err != nil {
				return fmt.Errorf("failed to build model: %w", err)
			}

			trainer := train.NewTrainer(m, lr)
			
			cmd.Printf("Starting training for %d epochs (LR: %f)...\n", epochs, lr)
			trainer.Train(ds, epochs, extractor)

			cmd.Printf("Saving model to %s...\n", out)
			if err := model.SaveModel(out, m); err != nil {
				return fmt.Errorf("failed to save model: %w", err)
			}

			cmd.Println("Training complete!")
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
