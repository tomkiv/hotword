package model

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const (
	MagicBytes = "HWMD"
	VersionV1  = uint16(1)
	VersionV2  = uint16(2)
)

const (
	LayerTypeConv2D    = uint32(1)
	LayerTypeReLU      = uint32(2)
	LayerTypeSigmoid   = uint32(3)
	LayerTypeMaxPool2D = uint32(4)
	LayerTypeDense     = uint32(5)
	LayerTypeGRU       = uint32(6)
	LayerTypeLSTM      = uint32(7)
)

func layerToID(l Layer) uint32 {
	switch l.Type() {
	case "conv2d":
		return LayerTypeConv2D
	case "relu":
		return LayerTypeReLU
	case "sigmoid":
		return LayerTypeSigmoid
	case "maxpool2d":
		return LayerTypeMaxPool2D
	case "dense":
		return LayerTypeDense
	case "gru":
		return LayerTypeGRU
	case "lstm":
		return LayerTypeLSTM
	default:
		return 0
	}
}

// SaveModel saves a Model to a file using the Version 2 format.
func SaveModel(path string, m Model) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create model file: %w", err)
	}
	defer f.Close()

	// 1. Magic Bytes
	if _, err := f.Write([]byte(MagicBytes)); err != nil {
		return err
	}

	// 2. Version
	if err := binary.Write(f, binary.LittleEndian, VersionV2); err != nil {
		return err
	}

	layers := m.GetLayers()
	// 3. Layer Count
	if err := binary.Write(f, binary.LittleEndian, uint32(len(layers))); err != nil {
		return err
	}

	// 4. Layers
	for _, l := range layers {
		typeID := layerToID(l)
		if err := binary.Write(f, binary.LittleEndian, typeID); err != nil {
			return err
		}

		switch typeID {
		case LayerTypeConv2D:
			conv := l.(*Conv2DLayer)
			if err := saveTensor(f, conv.Weights); err != nil {
				return err
			}
			if err := saveBias(f, conv.Bias); err != nil {
				return err
			}
			if err := binary.Write(f, binary.LittleEndian, uint32(conv.Stride)); err != nil {
				return err
			}
			if err := binary.Write(f, binary.LittleEndian, uint32(conv.Padding)); err != nil {
				return err
			}

		case LayerTypeReLU, LayerTypeSigmoid:
			// No extra params

		case LayerTypeMaxPool2D:
			mp := l.(*MaxPool2DLayer)
			if err := binary.Write(f, binary.LittleEndian, uint32(mp.KernelSize)); err != nil {
				return err
			}
			if err := binary.Write(f, binary.LittleEndian, uint32(mp.Stride)); err != nil {
				return err
			}

		case LayerTypeDense:
			dense := l.(*DenseLayer)
			if err := saveTensor(f, dense.Weights); err != nil {
				return err
			}
			if err := saveBias(f, dense.Bias); err != nil {
				return err
			}

		case LayerTypeGRU, LayerTypeLSTM:
			var inputSize, hiddenSize int
			var weights []*Tensor
			var biases [][]float32

			if typeID == LayerTypeGRU {
				gru := l.(*GRULayer)
				inputSize, hiddenSize = gru.InputSize, gru.HiddenSize
				weights = []*Tensor{gru.Wz, gru.Wr, gru.Wh, gru.Uz, gru.Ur, gru.Uh}
				biases = [][]float32{gru.Bz, gru.Br, gru.Bh}
			} else {
				lstm := l.(*LSTMLayer)
				inputSize, hiddenSize = lstm.InputSize, lstm.HiddenSize
				weights = []*Tensor{lstm.Wi, lstm.Wf, lstm.Wo, lstm.Wg, lstm.Ui, lstm.Uf, lstm.Uo, lstm.Ug}
				biases = [][]float32{lstm.Bi, lstm.Bf, lstm.Bo, lstm.Bg}
			}

			if err := binary.Write(f, binary.LittleEndian, uint32(inputSize)); err != nil {
				return err
			}
			if err := binary.Write(f, binary.LittleEndian, uint32(hiddenSize)); err != nil {
				return err
			}
			for _, w := range weights {
				if err := saveTensor(f, w); err != nil {
					return err
				}
			}
			for _, b := range biases {
				if err := saveBias(f, b); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

// LoadModel loads a Model from a file.
func LoadModel(path string) (Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open model file: %w", err)
	}
	defer f.Close()

	// 1. Magic Bytes
	magic := make([]byte, 4)
	if _, err := io.ReadFull(f, magic); err != nil {
		return nil, err
	}
	if string(magic) != MagicBytes {
		return nil, fmt.Errorf("invalid magic bytes")
	}

	// 2. Version
	var version uint16
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, err
	}

	if version == VersionV1 {
		// Legacy V1 loader (simple weights/bias)
		w, b, err := loadLegacyV1(f)
		if err != nil {
			return nil, err
		}
		// Convert to SequentialModel
		return NewSequentialModel(NewDenseLayer(w, b), NewSigmoidLayer()), nil
	}

	if version != VersionV2 {
		return nil, fmt.Errorf("unsupported model version: %d", version)
	}

	// 3. Layer Count
	var layerCount uint32
	if err := binary.Read(f, binary.LittleEndian, &layerCount); err != nil {
		return nil, err
	}

	var layers []Layer
	for i := uint32(0); i < layerCount; i++ {
		var typeID uint32
		if err := binary.Read(f, binary.LittleEndian, &typeID); err != nil {
			return nil, err
		}

		var l Layer
		switch typeID {
		case LayerTypeConv2D:
			w, err := loadTensor(f)
			if err != nil {
				return nil, err
			}
			b, err := loadBias(f)
			if err != nil {
				return nil, err
			}
			var stride, padding uint32
			binary.Read(f, binary.LittleEndian, &stride)
			binary.Read(f, binary.LittleEndian, &padding)
			l = NewConv2DLayer(w, b, int(stride), int(padding))

		case LayerTypeReLU:
			l = NewReLULayer()
		case LayerTypeSigmoid:
			l = NewSigmoidLayer()

		case LayerTypeMaxPool2D:
			var kernelSize, stride uint32
			binary.Read(f, binary.LittleEndian, &kernelSize)
			binary.Read(f, binary.LittleEndian, &stride)
			l = NewMaxPool2DLayer(int(kernelSize), int(stride))

		case LayerTypeDense:
			w, err := loadTensor(f)
			if err != nil {
				return nil, err
			}
			b, err := loadBias(f)
			if err != nil {
				return nil, err
			}
			l = NewDenseLayer(w, b)

		case LayerTypeGRU, LayerTypeLSTM:
			var inputSize, hiddenSize uint32
			binary.Read(f, binary.LittleEndian, &inputSize)
			binary.Read(f, binary.LittleEndian, &hiddenSize)

			var layer Layer
			if typeID == LayerTypeGRU {
				gru := NewGRULayer(int(inputSize), int(hiddenSize))
				weights := []*Tensor{gru.Wz, gru.Wr, gru.Wh, gru.Uz, gru.Ur, gru.Uh}
				for i := range weights {
					w, err := loadTensor(f)
					if err != nil {
						return nil, err
					}
					copy(weights[i].Data, w.Data)
				}
				bz, _ := loadBias(f)
				br, _ := loadBias(f)
				bh, _ := loadBias(f)
				copy(gru.Bz, bz)
				copy(gru.Br, br)
				copy(gru.Bh, bh)
				layer = gru
			} else {
				lstm := NewLSTMLayer(int(inputSize), int(hiddenSize))
				weights := []*Tensor{lstm.Wi, lstm.Wf, lstm.Wo, lstm.Wg, lstm.Ui, lstm.Uf, lstm.Uo, lstm.Ug}
				for i := range weights {
					w, err := loadTensor(f)
					if err != nil {
						return nil, err
					}
					copy(weights[i].Data, w.Data)
				}
				bi, _ := loadBias(f)
				bf, _ := loadBias(f)
				bo, _ := loadBias(f)
				bg, _ := loadBias(f)
				copy(lstm.Bi, bi)
				copy(lstm.Bf, bf)
				copy(lstm.Bo, bo)
				copy(lstm.Bg, bg)
				layer = lstm
			}
			l = layer

		default:
			return nil, fmt.Errorf("unknown layer type ID: %d", typeID)
		}
		layers = append(layers, l)
	}

	return NewSequentialModel(layers...), nil
}

// Helpers

func saveTensor(w io.Writer, t *Tensor) error {
	if err := binary.Write(w, binary.LittleEndian, uint32(len(t.Shape))); err != nil {
		return err
	}
	for _, dim := range t.Shape {
		if err := binary.Write(w, binary.LittleEndian, uint32(dim)); err != nil {
			return err
		}
	}
	for _, val := range t.Data {
		if err := binary.Write(w, binary.LittleEndian, val); err != nil {
			return err
		}
	}
	return nil
}

func loadTensor(r io.Reader) (*Tensor, error) {
	var numDims uint32
	if err := binary.Read(r, binary.LittleEndian, &numDims); err != nil {
		return nil, err
	}
	shape := make([]int, numDims)
	size := 1
	for i := range shape {
		var dim uint32
		if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
			return nil, err
		}
		shape[i] = int(dim)
		size *= shape[i]
	}
	t := NewTensor(shape)
	for i := 0; i < size; i++ {
		var val float32
		if err := binary.Read(r, binary.LittleEndian, &val); err != nil {
			return nil, err
		}
		t.Data[i] = val
	}
	return t, nil
}

func saveBias(w io.Writer, b []float32) error {
	if err := binary.Write(w, binary.LittleEndian, uint32(len(b))); err != nil {
		return err
	}
	for _, val := range b {
		if err := binary.Write(w, binary.LittleEndian, val); err != nil {
			return err
		}
	}
	return nil
}

func loadBias(r io.Reader) ([]float32, error) {
	var length uint32
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return nil, err
	}
	b := make([]float32, length)
	for i := range b {
		var val float32
		if err := binary.Read(r, binary.LittleEndian, &val); err != nil {
			return nil, err
		}
		b[i] = val
	}
	return b, nil
}

func loadLegacyV1(r io.Reader) (*Tensor, []float32, error) {
	var numRows, numCols, numBias uint32
	if err := binary.Read(r, binary.LittleEndian, &numRows); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &numCols); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &numBias); err != nil {
		return nil, nil, err
	}

	weights := NewTensor([]int{int(numRows), int(numCols)})
	for i := range weights.Data {
		var val float32
		binary.Read(r, binary.LittleEndian, &val)
		weights.Data[i] = val
	}

	bias := make([]float32, numBias)
	for i := range bias {
		var val float32
		binary.Read(r, binary.LittleEndian, &val)
		bias[i] = val
	}

	return weights, bias, nil
}
