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
var trainStride int
var trainMaxLen int
var trainOnset bool
var trainAugProb float32
var trainMaxNoise float32
var trainMaxShift int
var trainMaxGain float32
var trainThreads int

// NewTrainCmd creates a new train command
func NewTrainCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "train",
		Short: "Train a hotword detection model",
		Long: `Train a hotword detection model from WAV samples in hotword and background directories.

Variable-length input options:
  --stride: Extract overlapping 1-second windows with the specified stride (samples).
            Example: --stride 8000 for 50% overlap at 16kHz.
  --max-len: Use variable-length mode with padding/masking up to max-len samples.
            Shorter audio is padded, longer audio truncated from end.
  --onset: Use onset detection to find where hotword starts in each file.
            Automatically crops audio from the detected onset position.

Augmentation options:
  --augment-prob: Probability of applying random augmentations to hotword samples (0.0 to 1.0).
  --max-noise: Maximum noise ratio to mix into samples (0.0 to 1.0).
  --max-shift: Maximum random time shift in milliseconds.
  --max-gain: Maximum random gain/volume scaling (e.g. 0.2 for 0.8x-1.2x).

Performance options:
  --threads: Number of CPU threads to use for parallel training (sharded SGD).
             Defaults to number of CPU cores. Set to 1 for sequential.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			data := viper.GetString("train.data")
			out := viper.GetString("train.out")
			epochs := viper.GetInt("train.epochs")
			lr := float32(viper.GetFloat64("train.lr"))
			stride := viper.GetInt("train.stride")
			maxLen := viper.GetInt("train.max_len")
			onset := viper.GetBool("train.onset")
			threads := viper.GetInt("train.threads")

			// Augmentation params
			augProb := float32(viper.GetFloat64("train.augment_prob"))
			maxNoise := float32(viper.GetFloat64("train.max_noise"))
			maxShift := viper.GetInt("train.max_shift")
			maxGain := float32(viper.GetFloat64("train.max_gain"))

			sampleRate := 16000
			windowSize := 512
			hopSize := 256
			numMelFilters := 40
			windowLen := 16000 // 1 second at 16kHz

			cmd.Printf("Loading dataset from %s...\n", data)
			hotwordDir := filepath.Join(data, "hotword")
			backgroundDir := filepath.Join(data, "background")

			var ds *train.Dataset
			var err error

			// Choose loading mode based on flags
			if onset && stride > 0 {
				// Combined: Onset detection + Window/Stride extraction
				cmd.Printf("Using onset detection + windowed loading (stride=%d samples, %.2fs)\n", stride, float64(stride)/float64(sampleRate))
				ds, err = train.LoadDatasetWithOnsetAndStride(hotwordDir, backgroundDir, windowLen, stride, sampleRate, 0.1)
			} else if onset {
				// Option 2: Onset detection mode only
				cmd.Printf("Using onset detection (threshold=0.1)\n")
				ds, err = train.LoadDatasetWithOnset(hotwordDir, backgroundDir, windowLen, sampleRate, 0.1)
			} else if stride > 0 {
				// Option 1: Window/Stride extraction mode
				cmd.Printf("Using windowed loading (stride=%d samples, %.2fs)\n", stride, float64(stride)/float64(sampleRate))
				ds, err = train.LoadDatasetWindowed(hotwordDir, backgroundDir, windowLen, stride)
			} else if maxLen > 0 {
				// Option 3: Variable length with padding mode
				cmd.Printf("Using padded loading (max_len=%d samples, %.2fs)\n", maxLen, float64(maxLen)/float64(sampleRate))
				ds, err = train.LoadDatasetWithPadding(hotwordDir, backgroundDir, maxLen)
			} else {
				// Legacy mode: first 1 second only
				ds, err = train.LoadDataset(hotwordDir, backgroundDir)
			}

			if err != nil {
				return fmt.Errorf("failed to load dataset: %w", err)
			}

			if len(ds.Samples) == 0 {
				return fmt.Errorf("no samples found in dataset")
			}

			cmd.Printf("Loaded %d samples\n", len(ds.Samples))

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

			var t train.AugmentorTrainer
			if threads > 1 || threads == 0 {
				t = train.NewParallelTrainer(m, lr, threads)
			} else {
				t = train.NewTrainer(m, lr)
			}

			// Setup dynamic augmentor if needed
			if augProb > 0 {
				cmd.Printf("Using dynamic augmentation (Prob: %.2f, MaxNoise: %.2f, MaxShift: %dms, MaxGain: %.2f)\n", augProb, maxNoise, maxShift, maxGain)
				
				// Use background samples from dataset as noise pool
				var noisePool []train.Sample
				for _, s := range ds.Samples {
					if !s.IsHotword {
						noisePool = append(noisePool, s)
					}
				}
				
				aug := train.NewAugmentor(train.AugmentorConfig{
					AugmentProb:   augProb,
					MaxNoiseRatio: maxNoise,
					MaxShiftMs:    maxShift,
					MaxGainScale:  maxGain,
				}, noisePool)
				t.SetAugmentor(aug)
			}

			cmd.Printf("Starting training for %d epochs (LR: %f)...\n", epochs, lr)
			t.Train(ds, epochs, extractor)

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
	cmd.Flags().IntVar(&trainStride, "stride", 0, "Window stride in samples for overlapping extraction (e.g., 8000 for 50% overlap)")
	cmd.Flags().IntVar(&trainMaxLen, "max-len", 0, "Max audio length in samples for padding mode (e.g., 32000 for 2 seconds)")
	cmd.Flags().BoolVar(&trainOnset, "onset", false, "Use onset detection to crop hotword files from where audio activity starts")
	cmd.Flags().Float32Var(&trainAugProb, "augment-prob", 0, "Probability of applying random augmentations to hotword samples")
	cmd.Flags().Float32Var(&trainMaxNoise, "max-noise", 0.2, "Maximum noise ratio to mix into samples")
	cmd.Flags().IntVar(&trainMaxShift, "max-shift", 100, "Maximum random time shift in milliseconds")
	cmd.Flags().Float32Var(&trainMaxGain, "max-gain", 0.1, "Maximum random gain/volume scaling")
	cmd.Flags().IntVar(&trainThreads, "threads", 0, "Number of CPU threads for parallel training (0 = use all cores)")

	viper.BindPFlag("train.data", cmd.Flags().Lookup("data"))
	viper.BindPFlag("train.out", cmd.Flags().Lookup("out"))
	viper.BindPFlag("train.epochs", cmd.Flags().Lookup("epochs"))
	viper.BindPFlag("train.lr", cmd.Flags().Lookup("lr"))
	viper.BindPFlag("train.stride", cmd.Flags().Lookup("stride"))
	viper.BindPFlag("train.max_len", cmd.Flags().Lookup("max-len"))
	viper.BindPFlag("train.onset", cmd.Flags().Lookup("onset"))
	viper.BindPFlag("train.augment_prob", cmd.Flags().Lookup("augment-prob"))
	viper.BindPFlag("train.max_noise", cmd.Flags().Lookup("max-noise"))
	viper.BindPFlag("train.max_shift", cmd.Flags().Lookup("max-shift"))
	viper.BindPFlag("train.max_gain", cmd.Flags().Lookup("max-gain"))
	viper.BindPFlag("train.threads", cmd.Flags().Lookup("threads"))

	return cmd
}

var trainCmd = NewTrainCmd()

func init() {
	rootCmd.AddCommand(trainCmd)
}
