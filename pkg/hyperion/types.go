package hyperion

import "time"

// OHLCV represents a candlestick bar
type OHLCV struct {
	Date   time.Time
	Open   float64
	High   float64
	Low    float64
	Close  float64
	Volume float64
}

// Features represents the feature vector for ML
type Features struct {
	Returns1     float64
	Returns5     float64
	Returns10    float64
	SMA10        float64
	SMA30        float64
	RSI          float64
	ATR          float64
	MACD         float64
	BollingerPos float64
	VolumeRatio  float64
	Label        int // 1 = up, 0 = down
}
