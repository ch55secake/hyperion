package indicators

// CalculateMACD computes MACD and Signal line
func CalculateMACD(prices []float64) (float64, float64) {
	if len(prices) < 26 {
		return 0, 0
	}

	ema12 := CalculateEMA(prices, 12)
	ema26 := CalculateEMA(prices, 26)
	macd := ema12 - ema26

	// Calculate signal line (9-period EMA of MACD)
	//macdValues := []float64{macd} // Simplified for demo
	signal := macd

	return macd, signal
}

// CalculateMomentum
// Momentum calculates the Momentum indicator for a given slice of prices.
// period = number of periods ago to compare against.
// Returns a slice of momentum values.
func CalculateMomentum(prices []float64, period int) []float64 {
	n := len(prices)
	if period <= 0 || n <= period {
		return nil
	}

	momentum := make([]float64, n-period)
	for i := period; i < n; i++ {
		momentum[i-period] = prices[i] - prices[i-period]
	}

	return momentum
}

// CalculateROC
// ROC calculates the Rate of Change (ROC) indicator for a given slice of prices.
// period = number of periods ago to compare against.
// Returns a slice of ROC values (in percentage).
func CalculateROC(prices []float64, period int) []float64 {
	n := len(prices)
	if period <= 0 || n <= period {
		return nil
	}

	roc := make([]float64, n-period)
	for i := period; i < n; i++ {
		prev := prices[i-period]
		if prev == 0 {
			roc[i-period] = 0
			continue
		}
		roc[i-period] = ((prices[i] - prev) / prev) * 100.0
	}

	return roc
}
