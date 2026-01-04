package train

import (
	"testing"
)

func TestProgressBar(t *testing.T) {
	pb := NewProgressBar(100, "Testing")
	if pb.Total != 100 {
		t.Errorf("Expected total 100, got %d", pb.Total)
	}
	if pb.Label != "Testing" {
		t.Errorf("Expected label 'Testing', got %s", pb.Label)
	}

	pb.Update(10)
	if pb.Current != 10 {
		t.Errorf("Expected current 10, got %d", pb.Current)
	}

	// Test progress percentage calculation (internal logic)
	pct := pb.GetPercentage()
	if pct != 10.0 {
		t.Errorf("Expected percentage 10.0, got %f", pct)
	}
}
