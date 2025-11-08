package data

import (
	"fmt"
	"log"
	"time"
)

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
	Returns3     float64
	Returns5     float64
	Returns10    float64
	SMA5Ratio    float64
	SMA20Ratio   float64
	EMA12Ratio   float64
	RSI          float64
	RSISlope     float64
	ATRRatio     float64
	MACD         float64
	MACDSignal   float64
	BollingerPos float64
	VolumeRatio  float64
	VolumeSMA    float64
	Momentum     float64
	ROC          float64 // Rate of Change
	StochK       float64 // Stochastic Oscillator
	Label        int     // 1 = up, 0 = down
	CCI          float64
}

func GetFeatureValue(f Features, idx int) float64 {
	switch idx {
	case 0:
		return f.Returns1
	case 1:
		return f.Returns3
	case 2:
		return f.Returns5
	case 3:
		return f.Returns10
	case 4:
		return f.SMA5Ratio
	case 5:
		return f.SMA20Ratio
	case 6:
		return f.EMA12Ratio
	case 7:
		return f.RSI
	case 8:
		return f.RSISlope
	case 9:
		return f.ATRRatio
	case 10:
		return f.MACD
	case 11:
		return f.MACDSignal
	case 12:
		return f.BollingerPos
	case 13:
		return f.VolumeRatio
	case 14:
		return f.VolumeSMA
	case 15:
		return f.Momentum
	case 16:
		return f.ROC
	case 17:
		return f.StochK
	default:
		return 0
	}
}

// RandomForest implementation
type RandomForest struct {
	Trees       []*DecisionTree
	NumTrees    int
	MaxDepth    int
	MinSample   int
	MaxFeatures int
}

func (rf *RandomForest) Predict(f Features) float64 {
	sum := 0.0
	for _, tree := range rf.Trees {
		sum += tree.predict(f)
	}
	return sum / float64(rf.NumTrees)
}

// DecisionTree node
type DecisionTree struct {
	FeatureIdx int
	Threshold  float64
	Left       *DecisionTree
	Right      *DecisionTree
	Prediction float64
	IsLeaf     bool
}

func (dt *DecisionTree) predict(f Features) float64 {
	if dt.IsLeaf {
		return dt.Prediction
	}

	if GetFeatureValue(f, dt.FeatureIdx) <= dt.Threshold {
		return dt.Left.predict(f)
	}
	return dt.Right.predict(f)
}

// TradingStrategy implements ML-based trading
type TradingStrategy struct {
	Data         []OHLCV
	Features     []Features
	TrainSize    int
	Model        *RandomForest
	Position     float64
	Cash         float64
	PortfolioVal float64
	InitialCash  float64
}

// NewTradingStrategy creates a new strategy
func NewTradingStrategy(data []OHLCV, initialCash float64) *TradingStrategy {
	return &TradingStrategy{
		Data:         data,
		Cash:         initialCash,
		InitialCash:  initialCash,
		PortfolioVal: initialCash,
		Position:     0,
	}
}

// Backtest with improved logic
func (ts *TradingStrategy) Backtest() {
	log.Printf("Backtesting with %d days of data...\n", len(ts.Data)-ts.TrainSize)
	testDataSize := int(float64(len(ts.Features)) * 0.7)
	fmt.Printf("Test data size: %d\n", testDataSize)
	testData := ts.Features[testDataSize:]
	testStartIdx := testDataSize + 50 // Offset for the initial period

	correct := 0
	total := 0
	trades := 0
	wins := 0
	losses := 0

	var equity []float64
	equity = append(equity, ts.Cash)

	fmt.Println("\n=== Backtesting Results ===")

	for i, feature := range testData {
		prediction := ts.Model.Predict(feature)
		fmt.Printf("Day %d: Prediction: %.1f%%\n", i, prediction*100)
		actualIdx := testStartIdx + i

		if actualIdx >= len(ts.Data)-1 {
			break
		}

		currentPrice := ts.Data[actualIdx].Close

		// More aggressive thresholds: 0.52 and 0.48 - works with more expensive stocks
		// Penny stock merchant 0.58 and 0.48
		if prediction > 0.58 && ts.Position == 0 {
			// Buy
			shares := ts.Cash / currentPrice
			ts.Position = shares
			ts.Cash = 0
			trades++
			fmt.Printf("Day %d: BUY at $%.2f (confidence: %.1f%%)\n", i, currentPrice, prediction*100)
		} else if prediction < 0.48 && ts.Position > 0 {
			// Sell
			ts.Cash = ts.Position * currentPrice
			profitLoss := ts.Cash - ts.InitialCash
			if profitLoss > 0 {
				wins++
			} else {
				losses++
			}
			ts.Position = 0
			trades++
			fmt.Printf("Day %d: SELL at $%.2f (confidence: %.1f%%), P/L: $%.2f\n",
				i, currentPrice, (1-prediction)*100, profitLoss)
		}

		// Calculate current equity
		currentEquity := ts.Cash
		if ts.Position > 0 {
			currentEquity = ts.Position * currentPrice
		}
		equity = append(equity, currentEquity)

		// Check accuracy
		if (prediction > 0.5 && feature.Label == 1) || (prediction <= 0.5 && feature.Label == 0) {
			correct++
		}
		total++
	}

	// Close any open position
	if ts.Position > 0 && testStartIdx+len(testData)-1 < len(ts.Data) {
		finalPrice := ts.Data[testStartIdx+len(testData)-1].Close
		ts.Cash = ts.Position * finalPrice
		ts.Position = 0
	}

	finalReturn := (ts.Cash - ts.InitialCash) / ts.InitialCash * 100
	accuracy := float64(correct) / float64(total) * 100
	winRate := 0.0
	if trades > 0 {
		winRate = float64(wins) / float64(wins+losses) * 100
	}

	fmt.Printf("\n=== Performance Metrics ===\n")
	fmt.Printf("Prediction Accuracy: %.2f%%\n", accuracy)
	fmt.Printf("Total Trades: %d\n", trades)
	fmt.Printf("Wins: %d, Losses: %d, Win Rate: %.1f%%\n", wins, losses, winRate)
	fmt.Printf("Initial Capital: $%.2f\n", ts.InitialCash)
	fmt.Printf("Final Capital: $%.2f\n", ts.Cash)
	fmt.Printf("Total Return: %.2f%%\n", finalReturn)

	// Calculate max drawdown
	maxEquity := equity[0]
	maxDrawdown := 0.0
	for _, eq := range equity {
		if eq > maxEquity {
			maxEquity = eq
		}
		drawdown := (maxEquity - eq) / maxEquity * 100
		if drawdown > maxDrawdown {
			maxDrawdown = drawdown
		}
	}
	fmt.Printf("Max Drawdown: %.2f%%\n", maxDrawdown)
}
