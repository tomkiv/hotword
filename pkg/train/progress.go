package train

import (
	"fmt"
	"strings"
	"time"
)

// ProgressBar handles dynamic CLI progress updates.
type ProgressBar struct {
	Total     int
	Current   int
	Label     string
	StartTime time.Time
	Width     int
}

// NewProgressBar creates a new ProgressBar instance.
func NewProgressBar(total int, label string) *ProgressBar {
	return &ProgressBar{
		Total:     total,
		Label:     label,
		StartTime: time.Now(),
		Width:     30,
	}
}

// Update sets the current progress and refreshes the terminal line.
func (p *ProgressBar) Update(current int) {
	p.Current = current
	p.render()
}

// GetPercentage returns the current progress as a percentage.
func (p *ProgressBar) GetPercentage() float64 {
	if p.Total == 0 {
		return 100.0
	}
	return (float64(p.Current) / float64(p.Total)) * 100.0
}

// Finish completes the progress bar and prints a newline.
func (p *ProgressBar) Finish() {
	p.Current = p.Total
	p.render()
	fmt.Println()
}

func (p *ProgressBar) render() {
	pct := p.GetPercentage()
	filled := int((float64(p.Width) * pct) / 100.0)
	if filled > p.Width {
		filled = p.Width
	}
	
	bar := strings.Repeat("=", filled)
	if filled < p.Width {
		bar += ">" + strings.Repeat(" ", p.Width-filled-1)
	}
	
elapsed := time.Since(p.StartTime)
	rate := float64(p.Current) / elapsed.Seconds()
	
	var eta string
	if p.Current > 0 && p.Current < p.Total {
		remaining := time.Duration(float64(p.Total-p.Current)/rate) * time.Second
		eta = fmt.Sprintf("ETA: %s", remaining.Round(time.Second))
	} else {
		eta = fmt.Sprintf("Elapsed: %s", elapsed.Round(time.Second))
	}

	fmt.Printf("\r%s [" + bar + "] %.1f%% (%d/%d) %.1f units/s %s\033[K", 
		p.Label, pct, p.Current, p.Total, rate, eta)
}
