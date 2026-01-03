//go:build linux

package capture

/*
#cgo LDFLAGS: -lasound
#include <alsa/asoundlib.h>

// Helper to open ALSA device
int open_pcm(snd_pcm_t **handle, const char *name, unsigned int rate) {
    int err;
    if ((err = snd_pcm_open(handle, name, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        return err;
    }
    if ((err = snd_pcm_set_params(*handle,
                                  SND_PCM_FORMAT_S16_LE,
                                  SND_PCM_ACCESS_RW_INTERLEAVED,
                                  1, // channels
                                  rate,
                                  1, // resample
                                  500000)) < 0) { // 0.5s latency
        snd_pcm_close(*handle);
        return err;
    }
    return 0;
}
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

type linuxDevice struct {
	handle     *C.snd_pcm_t
	sampleRate int
	buffer     []int16
	chunkSize  int
}

func Open(deviceName string, sampleRate int) (Device, error) {
	var handle *C.snd_pcm_t
	cName := C.CString(deviceName)
	defer C.free(unsafe.Pointer(cName))

	res := C.open_pcm(&handle, cName, C.uint(sampleRate))
	if res < 0 {
		return nil, fmt.Errorf("failed to open ALSA device %s: %s", deviceName, C.GoString(C.snd_strerror(res)))
	}

	chunkSize := 512 // Default chunk size
	return &linuxDevice{
		handle:     handle,
		sampleRate: sampleRate,
		buffer:     make([]int16, chunkSize),
		chunkSize:  chunkSize,
	}, nil
}

func (d *linuxDevice) Read() ([]float32, error) {
	if d.handle == nil {
		return nil, ErrDeviceClosed
	}

	frames := C.snd_pcm_readi(d.handle, unsafe.Pointer(&d.buffer[0]), C.snd_pcm_uframes_t(d.chunkSize))
	if frames < 0 {
		// Attempt to recover from xrun
		C.snd_pcm_prepare(d.handle)
		return nil, fmt.Errorf("ALSA read error: %s", C.GoString(C.snd_strerror(C.int(frames))))
	}

	// Convert int16 to float32
	out := make([]float32, frames)
	for i := 0; i < int(frames); i++ {
		out[i] = float32(d.buffer[i]) / 32768.0
	}

	return out, nil
}

func (d *linuxDevice) Close() error {
	if d.handle != nil {
		C.snd_pcm_close(d.handle)
		d.handle = nil
	}
	return nil
}
