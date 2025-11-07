package indicators

import "math"

// mean returns the average of a slice of float64s.
func mean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	var sum float64
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

// stdDev calculates the population standard deviation of a slice of float64.
func stdDev(data []float64, mean float64) float64 {
	if len(data) == 0 {
		return 0
	}
	var variance float64
	for _, v := range data {
		diff := v - mean
		variance += diff * diff
	}
	return math.Sqrt(variance / float64(len(data)))
}

// CalculateStd calculates the population standard deviatio
func CalculateStd(prices []float64, period int) float64 {
	if len(prices) < period {
		return 0
	}

	mean := CalculateSMA(prices, period)
	variance := 0.0

	for i := len(prices) - period; i < len(prices); i++ {
		diff := prices[i] - mean
		variance += diff * diff
	}

	return math.Sqrt(variance / float64(period))
}
