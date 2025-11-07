package indicators

// CalculateEMA computes exponential moving average
func CalculateEMA(prices []float64, period int) float64 {
	if len(prices) < period {
		return prices[len(prices)-1]
	}

	multiplier := 2.0 / float64(period+1)
	ema := CalculateSMA(prices[:period], period)

	for i := period; i < len(prices); i++ {
		ema = (prices[i] * multiplier) + (ema * (1 - multiplier))
	}

	return ema
}

// CalculateSMA computes simple moving average
func CalculateSMA(prices []float64, period int) float64 {
	if len(prices) < period {
		return 0
	}
	sum := 0.0
	for i := len(prices) - period; i < len(prices); i++ {
		sum += prices[i]
	}
	return sum / float64(period)
}

// CalculateWMA computes weighted moving average
func CalculateWMA(prices []float64, period int) []float64 {
	if period <= 0 || len(prices) < period {
		return nil
	}

	weightMovingAverages := make([]float64, len(prices)-period+1)
	denominator := float64((period * (period + 1)) / 2)

	for i := 0; i <= len(prices)-period; i++ {
		var weightedSum float64
		for j := 0; j < period; j++ {
			weight := float64(j + 1)
			weightedSum += prices[i+j] * weight
		}
		weightMovingAverages[i] = weightedSum / denominator
	}

	return weightMovingAverages
}
