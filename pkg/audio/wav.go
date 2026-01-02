package audio

import (
	"encoding/binary"
	"fmt"
	"io"
)

// LoadWAV reads a 16-bit PCM WAV file and returns the samples as float32
// normalized to [-1.0, 1.0]. It also returns the sample rate.
func LoadWAV(r io.Reader) ([]float32, int, error) {
	var header [12]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, 0, fmt.Errorf("failed to read RIFF header: %w", err)
	}

	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("invalid WAV file format")
	}

	var sampleRate int
	var numChannels int
	var samples []float32

	for {
		var chunkHeader [8]byte
		if _, err := io.ReadFull(r, chunkHeader[:]); err != nil {
			if err == io.EOF {
				break
			}
			return nil, 0, fmt.Errorf("failed to read chunk header: %w", err)
		}

		chunkID := string(chunkHeader[0:4])
		chunkSize := binary.LittleEndian.Uint32(chunkHeader[4:8])

		switch chunkID {
		case "fmt ":
			if chunkSize < 16 {
				return nil, 0, fmt.Errorf("invalid fmt chunk size")
			}
			var format uint16
			binary.Read(r, binary.LittleEndian, &format)
			if format != 1 { // PCM
				return nil, 0, fmt.Errorf("unsupported audio format: %d (only PCM supported)", format)
			}
			var channels uint16
			binary.Read(r, binary.LittleEndian, &channels)
			numChannels = int(channels)

			var sr uint32
			binary.Read(r, binary.LittleEndian, &sr)
			sampleRate = int(sr)

			// Skip ByteRate, BlockAlign, BitsPerSample
			io.CopyN(io.Discard, r, int64(chunkSize-8))
		case "data":
			numSamples := int(chunkSize) / (numChannels * 2)
			samples = make([]float32, numSamples)
			for i := 0; i < numSamples; i++ {
				var sum float32
				for c := 0; c < numChannels; c++ {
					var sample int16
					if err := binary.Read(r, binary.LittleEndian, &sample); err != nil {
						return nil, 0, fmt.Errorf("failed to read sample: %w", err)
					}
					sum += float32(sample) / 32768.0
				}
				samples[i] = sum / float32(numChannels)
			}
		default:
			// Skip unknown chunks
			if _, err := io.CopyN(io.Discard, r, int64(chunkSize)); err != nil {
				return nil, 0, fmt.Errorf("failed to skip chunk %s: %w", chunkID, err)
			}
		}
	}

	if sampleRate == 0 {
		return nil, 0, fmt.Errorf("no sample rate found in WAV file")
	}

	return samples, sampleRate, nil
}