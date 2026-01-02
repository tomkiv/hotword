package model

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const (
	MagicBytes = "HWMD"
	Version    = uint16(1)
)

// SaveModelToWriter serializes the model weights and biases to an io.Writer.
func SaveModelToWriter(w io.Writer, weights *Tensor, bias []float32) error {
	// 1. Magic Bytes
	if _, err := w.Write([]byte(MagicBytes)); err != nil {
		return fmt.Errorf("failed to write magic bytes: %w", err)
	}

	// 2. Version
	if err := binary.Write(w, binary.LittleEndian, Version); err != nil {
		return fmt.Errorf("failed to write version: %w", err)
	}

	// 3. Metadata
	// For this minimalist implementation, we'll store:
	// - numRows (num filters/units)
	// - numCols (input size)
	// - numBias
	numRows := uint32(weights.Shape[0])
	numCols := uint32(weights.Shape[1])
	numBias := uint32(len(bias))

	if err := binary.Write(w, binary.LittleEndian, numRows); err != nil {
		return fmt.Errorf("failed to write numRows: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, numCols); err != nil {
		return fmt.Errorf("failed to write numCols: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, numBias); err != nil {
		return fmt.Errorf("failed to write numBias: %w", err)
	}

	// 4. Weights Data
	for _, val := range weights.Data {
		if err := binary.Write(w, binary.LittleEndian, val); err != nil {
			return fmt.Errorf("failed to write weight value: %w", err)
		}
	}

	// 5. Bias Data
	for _, val := range bias {
		if err := binary.Write(w, binary.LittleEndian, val); err != nil {
			return fmt.Errorf("failed to write bias value: %w", err)
		}
	}

	return nil
}

// SaveModel saves the model weights and biases to a file.
func SaveModel(path string, weights *Tensor, bias []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create model file: %w", err)
	}
	defer f.Close()

	return SaveModelToWriter(f, weights, bias)
}
