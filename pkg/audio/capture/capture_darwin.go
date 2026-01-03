//go:build darwin

package capture

import (
	"encoding/binary"
	"fmt"
	"io"
	"os/exec"
)

type darwinDevice struct {
	cmd    *exec.Cmd
	stdout io.ReadCloser
	buffer []byte
}

func Open(deviceName string, sampleRate int) (Device, error) {
	// Try 'rec' from SoX first, then 'ffmpeg'
	var cmd *exec.Cmd
	
	// SoX command: rec -q -t raw -r 16000 -c 1 -b 16 -e signed-integer -
	if _, err := exec.LookPath("rec"); err == nil {
		cmd = exec.Command("rec", "-q", "-t", "raw", "-r", fmt.Sprintf("%d", sampleRate), "-c", "1", "-b", "16", "-e", "signed-integer", "-")
	} else if _, err := exec.LookPath("ffmpeg"); err == nil {
		// FFmpeg command: ffmpeg -f avfoundation -i ":0" -f s16le -ac 1 -ar 16000 -
		cmd = exec.Command("ffmpeg", "-hide_banner", "-loglevel", "panic", "-f", "avfoundation", "-i", ":0", "-f", "s16le", "-ac", "1", "-ar", fmt.Sprintf("%d", sampleRate), "-")
	} else {
		return nil, fmt.Errorf("neither 'rec' (SoX) nor 'ffmpeg' found in PATH. Please install one of them for macOS audio capture.")
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}

	if err := cmd.Start(); err != nil {
		return nil, err
	}

	return &darwinDevice{
		cmd:    cmd,
		stdout: stdout,
		buffer: make([]byte, 1024), // 512 samples * 2 bytes
	}, nil
}

func (d *darwinDevice) Read() ([]float32, error) {
	n, err := io.ReadFull(d.stdout, d.buffer)
	if err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return nil, nil
		}
		return nil, err
	}

	numSamples := n / 2
	out := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		sample := int16(binary.LittleEndian.Uint16(d.buffer[i*2 : i*2+2]))
		out[i] = float32(sample) / 32768.0
	}

	return out, nil
}

func (d *darwinDevice) Close() error {
	if d.cmd != nil && d.cmd.Process != nil {
		d.cmd.Process.Kill()
	}
	if d.stdout != nil {
		d.stdout.Close()
	}
	return nil
}
