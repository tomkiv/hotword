package train

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/tomkiv/hotword/pkg/model"
)

// ParallelTrainer manages training across multiple CPU cores by sharding the dataset.
type ParallelTrainer struct {
	masterModel *model.SequentialModel
	lr          float32
	numThreads  int
	augmentor   *Augmentor
}

// NewParallelTrainer creates a new ParallelTrainer.
func NewParallelTrainer(m *model.SequentialModel, lr float32, threads int) *ParallelTrainer {
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	return &ParallelTrainer{
		masterModel: m,
		lr:          lr,
		numThreads:  threads,
	}
}

// SetAugmentor sets the augmentor for the trainer.
func (p *ParallelTrainer) SetAugmentor(a *Augmentor) {
	p.augmentor = a
}

// Train runs the sharded training loop.
func (p *ParallelTrainer) Train(ds *Dataset, epochs int, featureExtractor func([]float32) *model.Tensor) {
	numSamples := len(ds.Samples)
	if numSamples == 0 {
		return
	}
	
	actualThreads := p.numThreads
	if numSamples < actualThreads {
		actualThreads = numSamples
	}
	
	shardSize := (numSamples + actualThreads - 1) / actualThreads

	fmt.Printf("Starting parallel training with %d threads (Shard size: %d)\n", actualThreads, shardSize)

	for epoch := 1; epoch <= epochs; epoch++ {
		var wg sync.WaitGroup
		var mu sync.Mutex
		var totalLoss float32
		var completedSamples int

		pb := NewProgressBar(numSamples, fmt.Sprintf("Epoch %d/%d", epoch, epochs))

		// 1. Create local model copies for each shard
		shardModels := make([]*model.SequentialModel, actualThreads)
		for i := 0; i < actualThreads; i++ {
			shardModels[i] = p.cloneModel(p.masterModel)
		}

		// 2. Launch workers
		for t := 0; t < actualThreads; t++ {
			wg.Add(1)
			go func(threadIdx int) {
				defer wg.Done()
				
				start := threadIdx * shardSize
				end := start + shardSize
				if end > numSamples {
					end = numSamples
				}
				if start >= end {
					return
				}

				trainer := NewTrainer(shardModels[threadIdx], p.lr)
				trainer.SetAugmentor(p.augmentor)

				var shardLoss float32
				shardSamples := ds.Samples[start:end]
				
				for _, sample := range shardSamples {
					audioData := sample.Audio
					if sample.IsHotword && p.augmentor != nil {
						audioData = p.augmentor.Augment(audioData)
					}
					
					features := featureExtractor(audioData)
					target := float32(0.0)
					if sample.IsHotword {
						target = 1.0
					}
					
					shardLoss += trainer.TrainStep(features, target)
					
					mu.Lock()
					completedSamples++
					pb.Update(completedSamples)
					mu.Unlock()
				}

				mu.Lock()
				totalLoss += shardLoss
				mu.Unlock()
			}(t)
		}
		wg.Wait()

		// 3. Weight Averaging
		p.averageWeights(shardModels)

		pb.Finish()
		fmt.Printf("Average Loss: %.4f\n", totalLoss/float32(numSamples))
	}
}

// cloneModel creates a deep copy of a sequential model.
func (p *ParallelTrainer) cloneModel(m *model.SequentialModel) *model.SequentialModel {
	layers := m.GetLayers()
	newLayers := make([]model.Layer, len(layers))
	
	for i, l := range layers {
		// Generic cloning using Params/SetParams and Type
		weights, bias := l.Params()
		var newLayer model.Layer
		
		// We need to create a NEW instance of the layer type.
		// For simplicity, we use a switch on Type().
		switch l.Type() {
		case "conv2d":
			orig := l.(*model.Conv2DLayer)
			newLayer = model.NewConv2DLayer(model.NewTensor(weights.Shape), make([]float32, len(bias)), orig.Stride, orig.Padding)
		case "dense":
			newLayer = model.NewDenseLayer(model.NewTensor(weights.Shape), make([]float32, len(bias)))
		case "gru":
			orig := l.(*model.GRULayer)
			newLayer = model.NewGRULayer(orig.InputSize, orig.HiddenSize)
		case "lstm":
			orig := l.(*model.LSTMLayer)
			newLayer = model.NewLSTMLayer(orig.InputSize, orig.HiddenSize)
		case "relu":
			newLayer = model.NewReLULayer()
		case "sigmoid":
			newLayer = model.NewSigmoidLayer()
		case "maxpool2d":
			orig := l.(*model.MaxPool2DLayer)
			newLayer = model.NewMaxPool2DLayer(orig.KernelSize, orig.Stride)
		}
		
		if weights != nil {
			wCopy := model.NewTensor(weights.Shape)
			copy(wCopy.Data, weights.Data)
			bCopy := make([]float32, len(bias))
			copy(bCopy, bias)
			newLayer.SetParams(wCopy, bCopy)
		}
		newLayers[i] = newLayer
	}
	
	return model.NewSequentialModel(newLayers...)
}

// averageWeights averages parameters across all shard models and updates the master model.
func (p *ParallelTrainer) averageWeights(shardModels []*model.SequentialModel) {
	numShards := float32(len(shardModels))
	masterLayers := p.masterModel.GetLayers()
	
	for i, masterLayer := range masterLayers {
		mWeights, mBias := masterLayer.Params()
		if mWeights == nil {
			continue
		}
		
		// Reset master weights to 0 for accumulation
		for j := range mWeights.Data {
			mWeights.Data[j] = 0
		}
		for j := range mBias {
			mBias[j] = 0
		}
		
		// Accumulate from all shards
		for _, sm := range shardModels {
			sWeights, sBias := sm.GetLayers()[i].Params()
			for j := range mWeights.Data {
				mWeights.Data[j] += sWeights.Data[j]
			}
			for j := range mBias {
				mBias[j] += sBias[j]
			}
		}
		
		// Average
		for j := range mWeights.Data {
			mWeights.Data[j] /= numShards
		}
		for j := range mBias {
			mBias[j] /= numShards
		}
		
		// Update master layer
		masterLayer.SetParams(mWeights, mBias)
	}
}