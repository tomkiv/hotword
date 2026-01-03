package model

import "fmt"

// ErrUnsupportedLayer is returned when an unknown layer type is encountered.
type ErrUnsupportedLayer struct {
	Type string
}

func (e ErrUnsupportedLayer) Error() string {
	return fmt.Sprintf("unsupported layer type: %s", e.Type)
}
