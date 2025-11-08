package indicators

import (
	"errors"
	"math"

	types "hyperion-go/pkg/data"
)

// CalculateRSI computes Relative Strength Index
func CalculateRSI(prices []float64, period int) float64 {
	if len(prices) < period+1 {
		return 50.0
	}

	gains := make([]float64, 0)
	losses := make([]float64, 0)

	for i := len(prices) - period; i < len(prices); i++ {
		change := prices[i] - prices[i-1]
		if change > 0 {
			gains = append(gains, change)
			losses = append(losses, 0)
		} else {
			gains = append(gains, 0)
			losses = append(losses, -change)
		}
	}

	avgGain := 0.0
	avgLoss := 0.0
	for i := 0; i < len(gains); i++ {
		avgGain += gains[i]
		avgLoss += losses[i]
	}
	avgGain /= float64(period)
	avgLoss /= float64(period)

	if avgLoss == 0 {
		return 100.0
	}

	rs := avgGain / avgLoss
	return 100.0 - (100.0 / (1.0 + rs))
}

// CalculateStochastic computes Stochastic Oscillator %K
func CalculateStochastic(data []types.OHLCV, period int) float64 {
	if len(data) < period {
		return 50.0
	}

	recentData := data[len(data)-period:]

	lowest := recentData[0].Low
	highest := recentData[0].High

	for _, bar := range recentData {
		if bar.Low < lowest {
			lowest = bar.Low
		}
		if bar.High > highest {
			highest = bar.High
		}
	}

	current := data[len(data)-1].Close

	if highest == lowest {
		return 50.0
	}

	return 100.0 * (current - lowest) / (highest - lowest)
}

// CalculateCCI
// CCI calculates the Commodity Channel Index (CCI) for a given dataset.
// Each input (high, low, close) must be of equal length.
// Returns a slice of CCI values for the given period.
func CalculateCCI(data []types.OHLCV, constant float64) (float64, error) {
	if len(data) == 0 {
		return 0, errors.New("no data provided")
	}

	// Calculate typical prices
	typicalPrices := make([]float64, len(data))
	for i, d := range data {
		typicalPrices[i] = (d.High + d.Low + d.Close) / 3
	}

	// Calculate SMA of typical prices
	smaTP := 0.0
	for _, tp := range typicalPrices {
		smaTP += tp
	}
	smaTP /= float64(len(data))

	// Calculate mean deviation
	meanDev := 0.0
	for _, tp := range typicalPrices {
		meanDev += math.Abs(tp - smaTP)
	}
	meanDev /= float64(len(data))

	// Calculate CCI for the last point
	if meanDev == 0 {
		return 0, errors.New("mean deviation is zero")
	}

	lastTP := typicalPrices[len(typicalPrices)-1]
	cci := (lastTP - smaTP) / (constant * meanDev)

	return cci, nil
}
