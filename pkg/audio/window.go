package audio

// SlidingWindow manages a buffer of audio samples and provides overlapping windows.
type SlidingWindow struct {
	windowSize int
	hopSize    int
	buffer     []float32
}

// NewSlidingWindow creates a new SlidingWindow.
func NewSlidingWindow(windowSize, hopSize int) *SlidingWindow {
	return &SlidingWindow{
		windowSize: windowSize,
		hopSize:    hopSize,
		buffer:     make([]float32, 0, windowSize*2),
	}
}

// AddSamples appends new samples to the internal buffer.
func (sw *SlidingWindow) AddSamples(samples []float32) {
	sw.buffer = append(sw.buffer, samples...)
}

// NextWindow returns the next available window and true, or nil and false if not enough data.
func (sw *SlidingWindow) NextWindow() ([]float32, bool) {
	if len(sw.buffer) < sw.windowSize {
		return nil, false
	}

	window := make([]float32, sw.windowSize)
	copy(window, sw.buffer[:sw.windowSize])

	// Remove processed samples based on hopSize
	sw.buffer = sw.buffer[sw.hopSize:]

	return window, true
}

// Reset clears the internal buffer.
func (sw *SlidingWindow) Reset() {
	sw.buffer = sw.buffer[:0]
}
