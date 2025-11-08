package features

import (
	types "hyperion-go/pkg/data"
	indicators2 "hyperion-go/pkg/indicators"
)

// ExtractFeatures generates enhanced features
func ExtractFeatures(ts *types.TradingStrategy) []types.Features {
	features := make([]types.Features, 0)

	for i := 50; i < len(ts.Data)-5; i++ { // Look ahead 5 days for label
		prices := make([]float64, i+1)
		for j := 0; j <= i; j++ {
			prices[j] = ts.Data[j].Close
		}

		currentPrice := ts.Data[i].Close
		futureReturn := (ts.Data[i+5].Close - currentPrice) / currentPrice

		// Calculate returns
		returns1 := (currentPrice - ts.Data[i-1].Close) / ts.Data[i-1].Close
		returns3 := (currentPrice - ts.Data[i-3].Close) / ts.Data[i-3].Close
		returns5 := (currentPrice - ts.Data[i-5].Close) / ts.Data[i-5].Close
		returns10 := (currentPrice - ts.Data[i-10].Close) / ts.Data[i-10].Close

		// Moving averages
		sma5 := indicators2.CalculateSMA(prices, 5)
		sma20 := indicators2.CalculateSMA(prices, 20)
		ema12 := indicators2.CalculateEMA(prices, 12)

		// RSI and slope
		rsi := indicators2.CalculateRSI(prices, 14)
		rsiPrev := indicators2.CalculateRSI(prices[:len(prices)-5], 14)
		rsiSlope := rsi - rsiPrev

		cci, _ := indicators2.CalculateCCI(ts.Data[:i+1], 0.015)

		// ATR
		atr := indicators2.CalculateATR(ts.Data[:i+1], 14)

		// MACD
		macd, signal := indicators2.CalculateMACD(prices)

		// Bollinger Bands position
		sma20bb := indicators2.CalculateSMA(prices, 20)
		std := indicators2.CalculateStd(prices, 20)
		bollingerPos := 0.0
		if std > 0 {
			bollingerPos = (currentPrice - sma20bb) / (2 * std)
		}

		// Volume analysis
		volumes := make([]float64, 20)
		for j := 0; j < 20; j++ {
			volumes[j] = ts.Data[i-19+j].Volume
		}
		avgVol := indicators2.CalculateSMA(volumes, 20)
		volumeRatio := 0.0
		volumeSMA := 0.0
		if avgVol > 0 {
			volumeRatio = ts.Data[i].Volume / avgVol
			volumeSMA = avgVol
		}

		// Momentum
		momentum := currentPrice - ts.Data[i-10].Close

		// Rate of Change
		roc := 0.0
		if ts.Data[i-10].Close > 0 {
			roc = 100.0 * (currentPrice - ts.Data[i-10].Close) / ts.Data[i-10].Close
		}

		// Stochastic
		stochK := indicators2.CalculateStochastic(ts.Data[:i+1], 14)

		// Label: 1 if future return > 1%, 0 otherwise
		label := 0
		if futureReturn > 0.01 { // 1% threshold
			label = 1
		}

		feature := types.Features{
			Returns1:     returns1 * 100,
			Returns3:     returns3 * 100,
			Returns5:     returns5 * 100,
			Returns10:    returns10 * 100,
			SMA5Ratio:    (currentPrice - sma5) / sma5 * 100,
			SMA20Ratio:   (currentPrice - sma20) / sma20 * 100,
			EMA12Ratio:   (currentPrice - ema12) / ema12 * 100,
			RSI:          rsi,
			RSISlope:     rsiSlope,
			ATRRatio:     atr / currentPrice * 100,
			MACD:         macd / currentPrice * 100,
			MACDSignal:   signal / currentPrice * 100,
			BollingerPos: bollingerPos,
			VolumeRatio:  volumeRatio,
			VolumeSMA:    volumeSMA,
			Momentum:     momentum / currentPrice * 100,
			ROC:          roc,
			StochK:       stochK,
			Label:        label,
			CCI:          cci,
		}

		features = append(features, feature)
	}

	return features
}
