package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
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

// RandomForest implementation
type RandomForest struct {
	Trees       []*DecisionTree
	NumTrees    int
	MaxDepth    int
	MinSample   int
	MaxFeatures int
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
func CalculateStochastic(data []OHLCV, period int) float64 {
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

// CalculateATR computes Average True Range
func CalculateATR(data []OHLCV, period int) float64 {
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

// ExtractFeatures generates enhanced features
func (ts *TradingStrategy) ExtractFeatures() {
	ts.Features = make([]Features, 0)

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
		sma5 := CalculateSMA(prices, 5)
		sma20 := CalculateSMA(prices, 20)
		ema12 := CalculateEMA(prices, 12)

		// RSI and slope
		rsi := CalculateRSI(prices, 14)
		rsiPrev := CalculateRSI(prices[:len(prices)-5], 14)
		rsiSlope := rsi - rsiPrev

		// ATR
		atr := CalculateATR(ts.Data[:i+1], 14)

		// MACD
		macd, signal := CalculateMACD(prices)

		// Bollinger Bands position
		sma20bb := CalculateSMA(prices, 20)
		std := calculateStd(prices, 20)
		bollingerPos := 0.0
		if std > 0 {
			bollingerPos = (currentPrice - sma20bb) / (2 * std)
		}

		// Volume analysis
		volumes := make([]float64, 20)
		for j := 0; j < 20; j++ {
			volumes[j] = ts.Data[i-19+j].Volume
		}
		avgVol := CalculateSMA(volumes, 20)
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
		stochK := CalculateStochastic(ts.Data[:i+1], 14)

		// Label: 1 if future return > 1%, 0 otherwise
		label := 0
		if futureReturn > 0.01 { // 1% threshold
			label = 1
		}

		feature := Features{
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
		}

		ts.Features = append(ts.Features, feature)
	}
}

// Train with improved Random Forest
func (ts *TradingStrategy) Train(numTrees, maxDepth, minSample int) {
	ts.TrainSize = int(float64(len(ts.Features)) * 0.7)
	trainData := ts.Features[:ts.TrainSize]

	// Check label distribution
	ones := 0
	for _, f := range trainData {
		if f.Label == 1 {
			ones++
		}
	}
	fmt.Printf("Training data - Up: %d (%.1f%%), Down: %d (%.1f%%)\n",
		ones, float64(ones)/float64(len(trainData))*100,
		len(trainData)-ones, float64(len(trainData)-ones)/float64(len(trainData))*100)

	ts.Model = &RandomForest{
		NumTrees:    numTrees,
		MaxDepth:    maxDepth,
		MinSample:   minSample,
		MaxFeatures: 8, // sqrt(18) ≈ 4, but use more for better splits
		Trees:       make([]*DecisionTree, numTrees),
	}

	fmt.Printf("Training Random Forest with %d trees...\n", numTrees)

	rand.Seed(time.Now().UnixNano())

	for i := 0; i < numTrees; i++ {
		sample := bootstrapSample(trainData)
		ts.Model.Trees[i] = buildTree(sample, 0, maxDepth, minSample, ts.Model.MaxFeatures)
		if (i+1)%5 == 0 {
			//fmt.Printf("  Trained %d/%d trees\n", i+1, numTrees)
		}
	}

	fmt.Println("Training complete!")
}

func bootstrapSample(data []Features) []Features {
	sample := make([]Features, len(data))
	for i := 0; i < len(data); i++ {
		idx := rand.Intn(len(data))
		sample[i] = data[idx]
	}
	return sample
}

func buildTree(data []Features, depth, maxDepth, minSample, maxFeatures int) *DecisionTree {
	if depth >= maxDepth || len(data) < minSample {
		return &DecisionTree{
			IsLeaf:     true,
			Prediction: calculateMajorityClass(data),
		}
	}

	// Check if all same label
	allSame := true
	firstLabel := data[0].Label
	for _, f := range data {
		if f.Label != firstLabel {
			allSame = false
			break
		}
	}

	if allSame {
		return &DecisionTree{
			IsLeaf:     true,
			Prediction: float64(firstLabel),
		}
	}

	bestFeature, bestThreshold, bestGini := findBestSplit(data, maxFeatures)

	if bestGini >= 0.49 { // No significant improvement
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
		Left:       buildTree(left, depth+1, maxDepth, minSample, maxFeatures),
		Right:      buildTree(right, depth+1, maxDepth, minSample, maxFeatures),
		IsLeaf:     false,
	}
}

func findBestSplit(data []Features, maxFeatures int) (int, float64, float64) {
	bestGini := 1.0
	bestFeature := 0
	bestThreshold := 0.0

	// Randomly select features
	numFeatures := 18
	features := rand.Perm(numFeatures)[:maxFeatures]

	for _, featureIdx := range features {
		values := make([]float64, len(data))
		for i, f := range data {
			values[i] = getFeatureValue(f, featureIdx)
		}

		sort.Float64s(values)

		// Try more split points
		for q := 1; q <= 9; q++ {
			idx := len(values) * q / 10
			if idx >= len(values) || idx == 0 {
				continue
			}
			threshold := values[idx]

			left, right := splitData(data, featureIdx, threshold)
			if len(left) < 5 || len(right) < 5 {
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

func getFeatureValue(f Features, idx int) float64 {
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

func splitData(data []Features, featureIdx int, threshold float64) ([]Features, []Features) {
	var left, right []Features

	for _, f := range data {
		if getFeatureValue(f, featureIdx) <= threshold {
			left = append(left, f)
		} else {
			right = append(right, f)
		}
	}

	return left, right
}

func calculateGini(left, right []Features) float64 {
	totalSize := float64(len(left) + len(right))
	giniLeft := giniImpurity(left)
	giniRight := giniImpurity(right)

	return (float64(len(left))/totalSize)*giniLeft + (float64(len(right))/totalSize)*giniRight
}

func giniImpurity(data []Features) float64 {
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

func calculateMajorityClass(data []Features) float64 {
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

func (rf *RandomForest) Predict(f Features) float64 {
	sum := 0.0
	for _, tree := range rf.Trees {
		sum += tree.predict(f)
	}
	return sum / float64(rf.NumTrees)
}

func (dt *DecisionTree) predict(f Features) float64 {
	if dt.IsLeaf {
		return dt.Prediction
	}

	if getFeatureValue(f, dt.FeatureIdx) <= dt.Threshold {
		return dt.Left.predict(f)
	}
	return dt.Right.predict(f)
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

// GenerateSyntheticData with more realistic patterns
func GenerateSyntheticData(days int) []OHLCV {
	data := make([]OHLCV, days)
	price := 100.0
	rand.Seed(time.Now().UnixNano())

	trend := 0.0

	for i := 0; i < days; i++ {
		// Add trend changes every 50 days
		if i%50 == 0 {
			trend = (rand.Float64() - 0.5) * 0.004
		}

		// Random walk with trend and mean reversion
		dailyReturn := trend + (rand.Float64()-0.5)*0.02
		price *= (1 + dailyReturn)

		volatility := price * 0.01
		high := price + rand.Float64()*volatility
		low := price - rand.Float64()*volatility
		open := low + rand.Float64()*(high-low)
		close := low + rand.Float64()*(high-low)

		data[i] = OHLCV{
			Date:   time.Now().AddDate(0, 0, -days+i),
			Open:   open,
			High:   high,
			Low:    low,
			Close:  close,
			Volume: 1000000 + rand.Float64()*500000,
		}
	}

	return data
}

func LoadDataFromCSV(filename string) ([]OHLCV, error) {
	log.Printf("Loading data from %s\n", filename)
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

	var data []OHLCV
	for i, record := range records {
		if i == 0 {
			continue
		}

		date, _ := time.Parse("2006-01-02", record[0])
		open, _ := strconv.ParseFloat(record[1], 64)
		high, _ := strconv.ParseFloat(record[2], 64)
		low, _ := strconv.ParseFloat(record[3], 64)
		close, _ := strconv.ParseFloat(record[4], 64)
		volume, _ := strconv.ParseFloat(record[5], 64)

		data = append(data, OHLCV{
			Date:   date,
			Open:   open,
			High:   high,
			Low:    low,
			Close:  close,
			Volume: volume,
		})
	}

	log.Println("Got data with size of", len(data))

	return data, nil
}

func main() {
	fmt.Println("Enhanced ML Trading Strategy in Go")
	fmt.Println("===================================\n")

	resources := []string{
		//"resources/AXP_1y.csv",
		//"resources/RRR.L_1y.csv",
		//"resources/RRR_1y.csv",
		//"resources/NFLX_1y.csv",
		//"resources/AAPL_1y.csv",
		//"resources/GOOGL_1y.csv",
		//"resources/MSFT_1y.csv",
		//"resources/NVDA_1y.csv",
		//"resources/TSLA_1y.csv",
		//"resources/WORX_1y.csv",
		//"resources/BTC-USD_1y.csv",
		//"resources/AGL_1y.csv",
		//"resources/YGMZ_1y.csv",
		//"resources/BYND_1y.csv",
		"resources/IBRX_1y.csv",
		//"resources/EQX_1y.csv",
		//"resources/80M.L_1y.csv",
		//"resources/80M.L_5y.csv",
		//"resources/CNM_1y.csv",
		//"resources/AMZN_1y.csv",
	}

	// Generate more data for better training
	for _, resource := range resources {
		data, err := LoadDataFromCSV(resource)
		if err != nil {
			fmt.Println("Failed to load data from CSV")
		}

		fmt.Printf("Loaded %d days of data\n", len(data))

		strategy := NewTradingStrategy(data, 20000.0)

		fmt.Println("Extracting enhanced features...")
		strategy.ExtractFeatures()
		fmt.Printf("Generated %d feature vectors with 18 features each\n", len(strategy.Features))

		// Train with more trees and better parameters
		// 2000, 200, 18 - the penny stock merchant
		strategy.Train(2000, 200, 18) // 50 trees, max depth 8, min samples 15

		fmt.Printf("Backtesting for %s", resource)
		strategy.Backtest()
	}

	//fmt.Println("\n=== Next Steps ===")
	//fmt.Println("1. Try different thresholds in backtest (currently 0.52/0.48)")
	//fmt.Println("2. Add position sizing based on confidence")
	//fmt.Println("3. Implement stop-loss and take-profit")
	//fmt.Println("4. Test with real market data")
	//fmt.Println("5. Add walk-forward optimization")
}
