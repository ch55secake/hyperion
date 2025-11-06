package main

import (
	"encoding/csv"
	"fmt"
	"github.com/ch55secake/hyperion/pkg/hyperion"
	"math"
	"os"
	"sort"
	"strconv"
	"time"
)

// TradingStrategy implements ML-based trading
type TradingStrategy struct {
	Data         []hyperion.OHLCV
	Features     []hyperion.Features
	TrainSize    int
	Model        *RandomForest
	Position     float64 // Current position: 1 = long, 0 = flat, -1 = short
	Cash         float64
	PortfolioVal float64
}

// RandomForest simple implementation
type RandomForest struct {
	Trees     []*DecisionTree
	NumTrees  int
	MaxDepth  int
	MinSample int
}

// DecisionTree simple decision tree
type DecisionTree struct {
	FeatureIdx int
	Threshold  float64
	Left       *DecisionTree
	Right      *DecisionTree
	Prediction float64
	IsLeaf     bool
}

// NewTradingStrategy creates a new strategy
func NewTradingStrategy(data []hyperion.OHLCV, initialCash float64) *TradingStrategy {
	return &TradingStrategy{
		Data:         data,
		Cash:         initialCash,
		PortfolioVal: initialCash,
		Position:     0,
	}
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

// CalculateRSI computes Relative Strength Index
func CalculateRSI(prices []float64, period int) float64 {
	if len(prices) < period+1 {
		return 50.0
	}

	gains := 0.0
	losses := 0.0

	for i := len(prices) - period; i < len(prices); i++ {
		change := prices[i] - prices[i-1]
		if change > 0 {
			gains += change
		} else {
			losses += -change
		}
	}

	avgGain := gains / float64(period)
	avgLoss := losses / float64(period)

	if avgLoss == 0 {
		return 100.0
	}

	rs := avgGain / avgLoss
	return 100.0 - (100.0 / (1.0 + rs))
}

// CalculateATR computes Average True Range
func CalculateATR(data []hyperion.OHLCV, period int) float64 {
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

// CalculateMACD computes Moving Average Convergence Divergence
func CalculateMACD(prices []float64) float64 {
	if len(prices) < 26 {
		return 0
	}

	ema12 := calculateEMA(prices, 12)
	ema26 := calculateEMA(prices, 26)

	return ema12 - ema26
}

func calculateEMA(prices []float64, period int) float64 {
	if len(prices) < period {
		return 0
	}

	multiplier := 2.0 / float64(period+1)
	ema := prices[len(prices)-period]

	for i := len(prices) - period + 1; i < len(prices); i++ {
		ema = (prices[i] * multiplier) + (ema * (1 - multiplier))
	}

	return ema
}

// ExtractFeatures generates features from raw OHLCV data
func (ts *TradingStrategy) ExtractFeatures() {
	ts.Features = make([]hyperion.Features, 0)

	for i := 30; i < len(ts.Data)-1; i++ {
		prices := make([]float64, i+1)
		for j := 0; j <= i; j++ {
			prices[j] = ts.Data[j].Close
		}

		currentPrice := ts.Data[i].Close
		futurePrice := ts.Data[i+1].Close

		// Calculate returns
		returns1 := (currentPrice - ts.Data[i-1].Close) / ts.Data[i-1].Close
		returns5 := (currentPrice - ts.Data[i-5].Close) / ts.Data[i-5].Close
		returns10 := (currentPrice - ts.Data[i-10].Close) / ts.Data[i-10].Close

		// Calculate technical indicators
		sma10 := CalculateSMA(prices, 10)
		sma30 := CalculateSMA(prices, 30)
		rsi := CalculateRSI(prices, 14)
		atr := CalculateATR(ts.Data[:i+1], 14)
		macd := CalculateMACD(prices)

		// Bollinger position
		sma20 := CalculateSMA(prices, 20)
		std := calculateStd(prices, 20)
		bollingerPos := 0.0
		if std > 0 {
			bollingerPos = (currentPrice - sma20) / (2 * std)
		}

		// Volume ratio
		avgVol := 0.0
		for j := i - 9; j <= i; j++ {
			avgVol += ts.Data[j].Volume
		}
		avgVol /= 10
		volumeRatio := 0.0
		if avgVol > 0 {
			volumeRatio = ts.Data[i].Volume / avgVol
		}

		// Label: 1 if price goes up, 0 if down
		label := 0
		if futurePrice > currentPrice*1.001 { // 0.1% threshold
			label = 1
		}

		feature := hyperion.Features{
			Returns1:     returns1,
			Returns5:     returns5,
			Returns10:    returns10,
			SMA10:        (currentPrice - sma10) / sma10,
			SMA30:        (currentPrice - sma30) / sma30,
			RSI:          rsi / 100.0,
			ATR:          atr / currentPrice,
			MACD:         macd / currentPrice,
			BollingerPos: bollingerPos,
			VolumeRatio:  volumeRatio,
			Label:        label,
		}

		ts.Features = append(ts.Features, feature)
	}
}

func calculateStd(prices []float64, period int) float64 {
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

// Train trains the Random Forest model
func (ts *TradingStrategy) Train(numTrees, maxDepth, minSample int) {
	ts.TrainSize = int(float64(len(ts.Features)) * 0.7) // 70% train, 30% test
	trainData := ts.Features[:ts.TrainSize]

	ts.Model = &RandomForest{
		NumTrees:  numTrees,
		MaxDepth:  maxDepth,
		MinSample: minSample,
		Trees:     make([]*DecisionTree, numTrees),
	}

	fmt.Printf("Training Random Forest with %d trees...\n", numTrees)

	for i := 0; i < numTrees; i++ {
		// Bootstrap sample
		sample := bootstrapSample(trainData)
		ts.Model.Trees[i] = buildTree(sample, 0, maxDepth, minSample)
	}

	fmt.Println("Training complete!")
}

func bootstrapSample(data []hyperion.Features) []hyperion.Features {
	sample := make([]hyperion.Features, len(data))
	for i := 0; i < len(data); i++ {
		idx := int(float64(len(data)) * float64(time.Now().UnixNano()%1000) / 1000.0)
		if idx >= len(data) {
			idx = len(data) - 1
		}
		sample[i] = data[idx]
	}
	return sample
}

func buildTree(data []hyperion.Features, depth, maxDepth, minSample int) *DecisionTree {
	if depth >= maxDepth || len(data) < minSample {
		return &DecisionTree{
			IsLeaf:     true,
			Prediction: calculateMajorityClass(data),
		}
	}

	// Find best split
	bestFeature, bestThreshold, bestGini := findBestSplit(data)

	if bestGini == 1.0 { // No good split found
		return &DecisionTree{
			IsLeaf:     true,
			Prediction: calculateMajorityClass(data),
		}
	}

	left, right := splitData(data, bestFeature, bestThreshold)

	if len(left) == 0 || len(right) == 0 {
		return &DecisionTree{
			IsLeaf:     true,
			Prediction: calculateMajorityClass(data),
		}
	}

	return &DecisionTree{
		FeatureIdx: bestFeature,
		Threshold:  bestThreshold,
		Left:       buildTree(left, depth+1, maxDepth, minSample),
		Right:      buildTree(right, depth+1, maxDepth, minSample),
		IsLeaf:     false,
	}
}

func findBestSplit(data []hyperion.Features) (int, float64, float64) {
	bestGini := 1.0
	bestFeature := 0
	bestThreshold := 0.0

	// Try each feature
	for featureIdx := 0; featureIdx < 10; featureIdx++ {
		values := make([]float64, len(data))
		for i, f := range data {
			values[i] = getFeatureValue(f, featureIdx)
		}

		sort.Float64s(values)

		// Try splits at quartiles
		for q := 1; q <= 3; q++ {
			idx := len(values) * q / 4
			if idx >= len(values) {
				continue
			}
			threshold := values[idx]

			left, right := splitData(data, featureIdx, threshold)
			if len(left) == 0 || len(right) == 0 {
				continue
			}

			gini := calculateGini(left, right)
			if gini < bestGini {
				bestGini = gini
				bestFeature = featureIdx
				bestThreshold = threshold
			}
		}
	}

	return bestFeature, bestThreshold, bestGini
}

func getFeatureValue(f hyperion.Features, idx int) float64 {
	switch idx {
	case 0:
		return f.Returns1
	case 1:
		return f.Returns5
	case 2:
		return f.Returns10
	case 3:
		return f.SMA10
	case 4:
		return f.SMA30
	case 5:
		return f.RSI
	case 6:
		return f.ATR
	case 7:
		return f.MACD
	case 8:
		return f.BollingerPos
	case 9:
		return f.VolumeRatio
	default:
		return 0
	}
}

func splitData(data []hyperion.Features, featureIdx int, threshold float64) ([]hyperion.Features, []hyperion.Features) {
	var left, right []hyperion.Features

	for _, f := range data {
		if getFeatureValue(f, featureIdx) <= threshold {
			left = append(left, f)
		} else {
			right = append(right, f)
		}
	}

	return left, right
}

func calculateGini(left, right []hyperion.Features) float64 {
	totalSize := float64(len(left) + len(right))
	giniLeft := giniImpurity(left)
	giniRight := giniImpurity(right)

	return (float64(len(left))/totalSize)*giniLeft + (float64(len(right))/totalSize)*giniRight
}

func giniImpurity(data []hyperion.Features) float64 {
	if len(data) == 0 {
		return 0
	}

	ones := 0
	for _, f := range data {
		if f.Label == 1 {
			ones++
		}
	}

	p1 := float64(ones) / float64(len(data))
	p0 := 1.0 - p1

	return 1.0 - (p1*p1 + p0*p0)
}

func calculateMajorityClass(data []hyperion.Features) float64 {
	ones := 0
	for _, f := range data {
		if f.Label == 1 {
			ones++
		}
	}

	if ones > len(data)/2 {
		return 1.0
	}
	return 0.0
}

// Predict makes a prediction using the Random Forest
func (rf *RandomForest) Predict(f hyperion.Features) float64 {
	sum := 0.0
	for _, tree := range rf.Trees {
		sum += tree.predict(f)
	}
	return sum / float64(rf.NumTrees)
}

func (dt *DecisionTree) predict(f hyperion.Features) float64 {
	if dt.IsLeaf {
		return dt.Prediction
	}

	if getFeatureValue(f, dt.FeatureIdx) <= dt.Threshold {
		return dt.Left.predict(f)
	}
	return dt.Right.predict(f)
}

// Backtest runs the strategy on test data
func (ts *TradingStrategy) Backtest() {
	testData := ts.Features[ts.TrainSize:]

	correct := 0
	total := 0
	trades := 0

	fmt.Println("\n=== Backtesting Results ===")

	for i, feature := range testData {
		prediction := ts.Model.Predict(feature)

		// Trading logic: buy if prediction > 0.6, sell if < 0.4
		if prediction > 0.6 && ts.Position == 0 {
			// Buy
			ts.Position = 1
			trades++
			fmt.Printf("Day %d: BUY signal (confidence: %.2f%%)\n", i, prediction*100)
		} else if prediction < 0.4 && ts.Position == 1 {
			// Sell
			ts.Position = 0
			trades++
			fmt.Printf("Day %d: SELL signal (confidence: %.2f%%)\n", i, (1-prediction)*100)
		}

		// Check accuracy
		if (prediction > 0.5 && feature.Label == 1) || (prediction <= 0.5 && feature.Label == 0) {
			correct++
		}
		total++
	}

	accuracy := float64(correct) / float64(total) * 100
	fmt.Printf("\nAccuracy: %.2f%%\n", accuracy)
	fmt.Printf("Total trades: %d\n", trades)
}

// LoadDataFromCSV loads OHLCV data from CSV file
func LoadDataFromCSV(filename string) ([]hyperion.OHLCV, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var data []hyperion.OHLCV
	for i, record := range records {
		if i == 0 { // Skip header
			continue
		}

		date, _ := time.Parse("2006-01-02", record[0])
		open, _ := strconv.ParseFloat(record[1], 64)
		high, _ := strconv.ParseFloat(record[2], 64)
		low, _ := strconv.ParseFloat(record[3], 64)
		close, _ := strconv.ParseFloat(record[4], 64)
		volume, _ := strconv.ParseFloat(record[5], 64)

		data = append(data, hyperion.OHLCV{
			Date:   date,
			Open:   open,
			High:   high,
			Low:    low,
			Close:  close,
			Volume: volume,
		})
	}

	return data, nil
}

// GenerateSyntheticData creates sample data for testing
func GenerateSyntheticData(days int) []hyperion.OHLCV {
	data := make([]hyperion.OHLCV, days)
	price := 100.0

	for i := 0; i < days; i++ {
		// Random walk with slight upward bias
		change := (float64(time.Now().UnixNano()%200)/100.0 - 0.95) * 2
		price += change

		volatility := 0.5
		data[i] = hyperion.OHLCV{
			Date:   time.Now().AddDate(0, 0, -days+i),
			Open:   price,
			High:   price + volatility,
			Low:    price - volatility,
			Close:  price,
			Volume: 1000000 + float64(time.Now().UnixNano()%500000),
		}
	}

	return data
}

func main() {
	fmt.Println("ML Trading Strategy in Go")
	fmt.Println("==========================\n")

	// Generate synthetic data (in production, load from CSV)
	data, err := LoadDataFromCSV("resources/sp500_1y.csv")
	if err != nil {
		fmt.Println("Error loading data from CSV")
	}

	fmt.Printf("Loaded %d days of data\n", len(data))

	// Create strategy
	strategy := NewTradingStrategy(data, 10000.0)

	// Extract features
	fmt.Println("Extracting features...")
	strategy.ExtractFeatures()
	fmt.Printf("Generated %d feature vectors\n", len(strategy.Features))

	// Train model
	strategy.Train(10, 5, 10) // 10 trees, max depth 5, min samples 10

	// Backtest
	strategy.Backtest()

	fmt.Println("\n=== Strategy Complete ===")
	fmt.Println("To use with real data:")
	fmt.Println("1. Save CSV with columns: Date,Open,High,Low,Close,Volume")
	fmt.Println("2. Replace GenerateSyntheticData() with LoadDataFromCSV()")
	fmt.Println("3. Add position sizing and risk management")
	fmt.Println("4. Implement live trading integration")
}
