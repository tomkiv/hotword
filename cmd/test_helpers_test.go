package cmd

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func createDummyData(t *testing.T) (string, func()) {
	tmpDir, err := os.MkdirTemp("", "hotword_cmd_test")
	if err != nil {
		t.Fatal(err)
	}
	hotwordDir := filepath.Join(tmpDir, "hotword")
	bgDir := filepath.Join(tmpDir, "background")
	os.MkdirAll(hotwordDir, 0755)
	os.MkdirAll(bgDir, 0755)

	createDummyWAV(filepath.Join(hotwordDir, "s.wav"))
	createDummyWAV(filepath.Join(bgDir, "b.wav"))

	return tmpDir, func() { os.RemoveAll(tmpDir) }
}

func createDummyWAV(path string) {
	f, _ := os.Create(path)
	defer f.Close()
	// Minimal 16-bit PCM WAV header + 1000 samples (enough for 512 window)
	numSamples := 1000
	f.Write([]byte("RIFF"))
	f.Write([]byte{0, 0, 0, 0}) // size
	f.Write([]byte("WAVEfmt "))
	f.Write([]byte{16, 0, 0, 0, 1, 0, 1, 0, 0x80, 0x3e, 0, 0, 0, 0, 0, 0, 2, 0, 16, 0})
	f.Write([]byte("data"))
	dataSize := uint32(numSamples * 2)
	binary.Write(f, binary.LittleEndian, dataSize)
	f.Write(make([]byte, dataSize))
}
