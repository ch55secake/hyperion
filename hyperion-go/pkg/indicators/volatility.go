package indicators

import (
	"math"

	types "hyperion-go/pkg/data"
)

// CalculateATR computes Average True Range
func CalculateATR(data []types.OHLCV, period int) float64 {
	if len(data) < period+1 {
		return 0
	}

	tr := 0.0
	for i := len(data) - period; i < len(data); i++ {
		high := data[i].High
		low := data[i].Low
		prevClose := data[i-1].Close

		trueRange := math.Max(high-low, math.Max(math.Abs(high-prevClose), math.Abs(low-prevClose)))
		tr += trueRange
	}

	return tr / float64(period)
}

// CalculateBollingerBands
// BollingerBands calculates the Bollinger Bands for a given price series.
// prices: slice of float64 closing prices
// period: number of periods for the SMA and standard deviation
// k: standard deviation multiplier (commonly 2)
// Returns three slices: upperBand, middleBand (SMA), lowerBand
func CalculateBollingerBands(prices []float64, period int, k float64) (upper, middle, lower []float64) {
	n := len(prices)
	if period <= 0 || n < period {
		return nil, nil, nil
	}

	middle = make([]float64, n-period+1)
	upper = make([]float64, n-period+1)
	lower = make([]float64, n-period+1)

	for i := 0; i <= n-period; i++ {
		window := prices[i : i+period]
		sma := mean(window)
		std := stdDev(window, sma)

		middle[i] = sma
		upper[i] = sma + k*std
		lower[i] = sma - k*std
	}

	return upper, middle, lower
}
