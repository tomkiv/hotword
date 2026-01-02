package audio

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func createWAV(numSamples int, sampleRate int, numChannels int) []byte {
	buf := new(bytes.Buffer)

	// RIFF Header
	buf.Write([]byte("RIFF"))
	binary.Write(buf, binary.LittleEndian, uint32(36+numSamples*2*numChannels))
	buf.Write([]byte("WAVE"))

	// fmt chunk
	buf.Write([]byte("fmt "))
	binary.Write(buf, binary.LittleEndian, uint32(16))
	binary.Write(buf, binary.LittleEndian, uint16(1)) // PCM
	binary.Write(buf, binary.LittleEndian, uint16(numChannels))
	binary.Write(buf, binary.LittleEndian, uint32(sampleRate))
	binary.Write(buf, binary.LittleEndian, uint32(sampleRate*numChannels*2)) // ByteRate
	binary.Write(buf, binary.LittleEndian, uint16(numChannels*2))           // BlockAlign
	binary.Write(buf, binary.LittleEndian, uint16(16))                     // BitsPerSample

	// data chunk
	buf.Write([]byte("data"))
	binary.Write(buf, binary.LittleEndian, uint32(numSamples*2*numChannels))
	for i := 0; i < numSamples*numChannels; i++ {
		binary.Write(buf, binary.LittleEndian, int16(i%32768))
	}

	return buf.Bytes()
}

func TestLoadWAV(t *testing.T) {
	t.Run("Valid Mono 16kHz", func(t *testing.T) {
		data := createWAV(1600, 16000, 1)
		samples, sampleRate, err := LoadWAV(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if sampleRate != 16000 {
			t.Errorf("Expected sample rate 16000, got %d", sampleRate)
		}
		if len(samples) != 1600 {
			t.Errorf("Expected 1600 samples, got %d", len(samples))
		}
	})

	t.Run("Valid Stereo 44.1kHz (Mixed to Mono)", func(t *testing.T) {
		data := createWAV(4410, 44100, 2)
		samples, sampleRate, err := LoadWAV(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if sampleRate != 44100 {
			t.Errorf("Expected sample rate 44100, got %d", sampleRate)
		}
		if len(samples) != 4410 {
			t.Errorf("Expected 4410 samples (mono-mixed), got %d", len(samples))
		}
	})

	t.Run("Invalid Header", func(t *testing.T) {
		data := []byte("NOTAWAVFILE")
		_, _, err := LoadWAV(bytes.NewReader(data))
		if err == nil {
			t.Error("Expected error for invalid header, got nil")
		}
	})
}
