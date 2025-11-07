package indicators

import (
	types "github.com/ch55secake/hyperion/pkg/data"
	"math"
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
func CalculateCCI(high, low, close []float64, period int) []float64 {
	n := len(close)
	if len(high) != n || len(low) != n || period <= 0 || n < period {
		return nil
	}

	typicalPrices := make([]float64, n)
	for i := 0; i < n; i++ {
		typicalPrices[i] = (high[i] + low[i] + close[i]) / 3.0
	}

	ccis := make([]float64, n-period+1)
	const constant = 0.015

	for i := 0; i <= n-period; i++ {
		window := typicalPrices[i : i+period]
		sma := mean(window)

		// Calculate mean deviation
		var md float64
		for _, tp := range window {
			md += math.Abs(tp - sma)
		}
		md /= float64(period)

		if md == 0 {
			ccis[i] = 0
			continue
		}

		// Current CCI value
		ccis[i] = (window[len(window)-1] - sma) / (constant * md)
	}

	return ccis
}
