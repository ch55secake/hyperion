package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Tree represents a single decision tree in the ensemble
type Tree struct {
	Feature   int     // Feature index for split (-1 for leaf)
	Threshold float64 // Split threshold
	Left      *Tree   // Left child
	Right     *Tree   // Right child
	LeafValue float64 // Prediction value for leaf nodes
	Gain      float64 // Information gain from this split
}

// XGBoost represents the gradient boosting model
type XGBoost struct {
	Trees          []*Tree
	LearningRate   float64
	MaxDepth       int
	MinChildWeight float64
	Lambda         float64 // L2 regularization
	Gamma          float64 // Minimum loss reduction for split
	Subsample      float64 // Row sampling ratio
	NumTrees       int
}

// StockData represents stock features and target
type StockData struct {
	Features [][]float64 // [samples][features]
	Target   []float64   // Price change or return
	Dates    []string    // Dates for reference
}

// YFinanceQuote represents a single price quote
type YFinanceQuote struct {
	Date   int64
	Open   float64
	High   float64
	Low    float64
	Close  float64
	Volume float64
}

// LoadStockDataFromCSV loads historical stock data from a CSV file
func LoadStockDataFromCSV(filepath string) ([]YFinanceQuote, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV: %v", err)
	}

	if len(records) < 2 {
		return nil, fmt.Errorf("CSV file is empty or has no data rows")
	}

	stockQuotes := make([]YFinanceQuote, 0)

	// Skip header row
	for i := 1; i < len(records); i++ {
		record := records[i]
		if len(record) < 6 {
			continue
		}

		// Parse date (format: YYYY-MM-DD)
		dateStr := strings.TrimSpace(record[0])
		t, err := time.Parse("2006-01-02", dateStr)
		if err != nil {
			continue
		}

		// Parse numeric values
		open, err1 := strconv.ParseFloat(strings.TrimSpace(record[1]), 64)
		high, err2 := strconv.ParseFloat(strings.TrimSpace(record[2]), 64)
		low, err3 := strconv.ParseFloat(strings.TrimSpace(record[3]), 64)
		closePrice, err4 := strconv.ParseFloat(strings.TrimSpace(record[4]), 64)
		volume, err5 := strconv.ParseFloat(strings.TrimSpace(record[5]), 64)

		// Skip rows with parsing errors or zero close price
		if err1 != nil || err2 != nil || err3 != nil || err4 != nil || err5 != nil || closePrice == 0 {
			continue
		}

		stockQuotes = append(stockQuotes, YFinanceQuote{
			Date:   t.Unix(),
			Open:   open,
			High:   high,
			Low:    low,
			Close:  closePrice,
			Volume: volume,
		})
	}

	if len(stockQuotes) == 0 {
		return nil, fmt.Errorf("no valid quotes found in CSV")
	}

	return stockQuotes, nil
}

// GetCSVFiles returns all CSV files in the historic_data directory
func GetCSVFiles(directory string) ([]string, error) {
	files, err := filepath.Glob(filepath.Join(directory, "*.csv"))
	if err != nil {
		return nil, fmt.Errorf("failed to read directory: %v", err)
	}
	return files, nil
}

// PrepareStockData converts raw quotes into features and targets
func PrepareStockData(quotes []YFinanceQuote) *StockData {
	if len(quotes) < 20 {
		return nil
	}

	features := make([][]float64, 0)
	targets := make([]float64, 0)
	dates := make([]string, 0)

	for i := 20; i < len(quotes)-1; i++ {
		// Calculate technical indicators

		// 1. Simple Moving Average (5 and 20 day)
		sma5 := calculateSMA(quotes, i, 5)
		sma20 := calculateSMA(quotes, i, 20)

		// 2. RSI (14 day)
		rsi := calculateRSI(quotes, i, 14)

		// 3. Price change percentage (1 day)
		priceChange1d := (quotes[i].Close - quotes[i-1].Close) / quotes[i-1].Close

		// 4. Price change percentage (5 day)
		priceChange5d := (quotes[i].Close - quotes[i-5].Close) / quotes[i-5].Close

		// 5. Volume change
		volumeChange := (quotes[i].Volume - quotes[i-1].Volume) / quotes[i-1].Volume

		// 6. Volatility (standard deviation of returns over 10 days)
		volatility := calculateVolatility(quotes, i, 10)

		// 7. High-Low range
		hlRange := (quotes[i].High - quotes[i].Low) / quotes[i].Close

		// Target: Next day's return
		target := (quotes[i+1].Close - quotes[i].Close) / quotes[i].Close

		feature := []float64{
			sma5,
			sma20,
			rsi,
			priceChange1d,
			priceChange5d,
			volumeChange,
			volatility,
			hlRange,
		}

		features = append(features, feature)
		targets = append(targets, target)
		dates = append(dates, time.Unix(quotes[i].Date, 0).Format("2006-01-02"))
	}

	return &StockData{
		Features: features,
		Target:   targets,
		Dates:    dates,
	}
}

// Technical indicator calculations
func calculateSMA(quotes []YFinanceQuote, idx, period int) float64 {
	sum := 0.0
	for i := idx - period + 1; i <= idx; i++ {
		sum += quotes[i].Close
	}
	return sum / float64(period)
}

func calculateRSI(quotes []YFinanceQuote, idx, period int) float64 {
	gains := 0.0
	losses := 0.0

	for i := idx - period + 1; i <= idx; i++ {
		change := quotes[i].Close - quotes[i-1].Close
		if change > 0 {
			gains += change
		} else {
			losses += -change
		}
	}

	avgGain := gains / float64(period)
	avgLoss := losses / float64(period)

	if avgLoss == 0 {
		return 100
	}

	rs := avgGain / avgLoss
	rsi := 100 - (100 / (1 + rs))
	return rsi
}

func calculateVolatility(quotes []YFinanceQuote, idx, period int) float64 {
	returns := make([]float64, period)
	for i := 0; i < period; i++ {
		returns[i] = (quotes[idx-period+i+1].Close - quotes[idx-period+i].Close) / quotes[idx-period+i].Close
	}

	mean := 0.0
	for _, r := range returns {
		mean += r
	}
	mean /= float64(period)

	variance := 0.0
	for _, r := range returns {
		diff := r - mean
		variance += diff * diff
	}
	variance /= float64(period)

	return math.Sqrt(variance)
}

// NewXGBoost creates a new XGBoost model
func NewXGBoost(numTrees int, learningRate float64, maxDepth int) *XGBoost {
	return &XGBoost{
		Trees:          make([]*Tree, 0),
		LearningRate:   learningRate,
		MaxDepth:       maxDepth,
		MinChildWeight: 0.5,
		Lambda:         1.0,
		Gamma:          0.0,
		Subsample:      1.0,
		NumTrees:       numTrees,
	}
}

// Train trains the XGBoost model on stock data
func (xgb *XGBoost) Train(data *StockData) {
	n := len(data.Target)

	// Initialize predictions with base score (mean)
	predictions := make([]float64, n)
	baseScore := mean(data.Target)
	for i := range predictions {
		predictions[i] = baseScore
	}

	// Add debug info
	fmt.Printf("Base score (mean of targets): %.8f\n", baseScore)
	fmt.Printf("Target standard deviation: %.8f\n", calculateStdDev(data.Target))

	// Build trees iteratively
	for t := 0; t < xgb.NumTrees; t++ {
		// Calculate gradients and hessians
		gradients := make([]float64, n)
		hessians := make([]float64, n)

		for i := 0; i < n; i++ {
			// For squared loss: gradient = prediction - actual
			gradients[i] = predictions[i] - data.Target[i]
			// For squared loss: hessian = 1
			hessians[i] = 1.0
		}

		// Sample rows if subsample < 1
		indices := xgb.sampleIndices(n)

		// Build tree
		tree := xgb.buildTree(data.Features, gradients, hessians, indices, 0)
		xgb.Trees = append(xgb.Trees, tree)

		// Update predictions
		for i := 0; i < n; i++ {
			pred := xgb.predictTree(tree, data.Features[i])
			predictions[i] += xgb.LearningRate * pred
		}

		// Calculate and print training loss
		loss := calculateMSE(predictions, data.Target)
		if t%10 == 0 {
			fmt.Printf("Iteration %d, MSE: %.8f\n", t, loss)
		}
	}
}

// buildTree recursively builds a decision tree
func (xgb *XGBoost) buildTree(features [][]float64, gradients, hessians []float64, indices []int, depth int) *Tree {
	// Check stopping conditions
	if depth >= xgb.MaxDepth || len(indices) == 0 {
		return xgb.createLeaf(gradients, hessians, indices)
	}

	// Calculate sum of gradients and hessians for this node
	G := sumByIndices(gradients, indices)
	H := sumByIndices(hessians, indices)

	if H < xgb.MinChildWeight {
		return xgb.createLeaf(gradients, hessians, indices)
	}

	// Find best split
	bestGain := -math.MaxFloat64
	bestFeature := -1
	bestThreshold := 0.0
	var bestLeftIndices, bestRightIndices []int

	numFeatures := len(features[0])

	for f := 0; f < numFeatures; f++ {
		// Get unique values for this feature
		values := make([]float64, len(indices))
		for i, idx := range indices {
			values[i] = features[idx][f]
		}
		sort.Float64s(values)

		// Try splits at midpoints between consecutive unique values
		for i := 0; i < len(values)-1; i++ {
			if values[i] == values[i+1] {
				continue
			}

			threshold := (values[i] + values[i+1]) / 2.0
			leftIndices, rightIndices := xgb.splitIndices(features, indices, f, threshold)

			if len(leftIndices) == 0 || len(rightIndices) == 0 {
				continue
			}

			// Calculate gain using XGBoost formula
			GL := sumByIndices(gradients, leftIndices)
			HL := sumByIndices(hessians, leftIndices)
			GR := sumByIndices(gradients, rightIndices)
			HR := sumByIndices(hessians, rightIndices)

			if HL < xgb.MinChildWeight || HR < xgb.MinChildWeight {
				continue
			}

			// Gain = 0.5 * [GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ)] - γ
			gain := 0.5*((GL*GL)/(HL+xgb.Lambda)+
				(GR*GR)/(HR+xgb.Lambda)-
				(G*G)/(H+xgb.Lambda)) - xgb.Gamma

			if gain > bestGain {
				bestGain = gain
				bestFeature = f
				bestThreshold = threshold
				bestLeftIndices = leftIndices
				bestRightIndices = rightIndices
			}
		}
	}

	// If no valid split found, create leaf
	if bestFeature == -1 || bestGain <= 0 {
		leaf := xgb.createLeaf(gradients, hessians, indices)
		if depth == 0 {
			fmt.Printf("Warning: Root node is a leaf with value: %.8f\n", leaf.LeafValue)
		}
		return leaf
	}

	// Create internal node and recursively build children
	tree := &Tree{
		Feature:   bestFeature,
		Threshold: bestThreshold,
		Gain:      bestGain,
	}

	tree.Left = xgb.buildTree(features, gradients, hessians, bestLeftIndices, depth+1)
	tree.Right = xgb.buildTree(features, gradients, hessians, bestRightIndices, depth+1)

	return tree
}

// createLeaf creates a leaf node with optimal weight
func (xgb *XGBoost) createLeaf(gradients, hessians []float64, indices []int) *Tree {
	G := sumByIndices(gradients, indices)
	H := sumByIndices(hessians, indices)

	// Optimal weight w* = -G/(H+λ)
	weight := -G / (H + xgb.Lambda)

	return &Tree{
		Feature:   -1, // -1 indicates leaf
		LeafValue: weight,
	}
}

// splitIndices splits indices based on feature and threshold
func (xgb *XGBoost) splitIndices(features [][]float64, indices []int, feature int, threshold float64) ([]int, []int) {
	var left, right []int
	for _, idx := range indices {
		if features[idx][feature] < threshold {
			left = append(left, idx)
		} else {
			right = append(right, idx)
		}
	}
	return left, right
}

// sampleIndices returns sampled indices based on subsample ratio
func (xgb *XGBoost) sampleIndices(n int) []int {
	if xgb.Subsample >= 1.0 {
		indices := make([]int, n)
		for i := 0; i < n; i++ {
			indices[i] = i
		}
		return indices
	}

	sampleSize := int(float64(n) * xgb.Subsample)
	indices := make([]int, sampleSize)
	for i := 0; i < sampleSize; i++ {
		indices[i] = i // Simplified: should use random sampling
	}
	return indices
}

// Predict makes predictions for new data
func (xgb *XGBoost) Predict(features [][]float64) []float64 {
	predictions := make([]float64, len(features))

	for i, feature := range features {
		prediction := 0.0
		for _, tree := range xgb.Trees {
			prediction += xgb.LearningRate * xgb.predictTree(tree, feature)
		}
		predictions[i] = prediction
	}

	return predictions
}

// predictTree makes a prediction using a single tree
func (xgb *XGBoost) predictTree(tree *Tree, feature []float64) float64 {
	if tree.Feature == -1 {
		return tree.LeafValue
	}

	if feature[tree.Feature] < tree.Threshold {
		return xgb.predictTree(tree.Left, feature)
	}
	return xgb.predictTree(tree.Right, feature)
}

// Helper functions
func mean(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func sumByIndices(values []float64, indices []int) float64 {
	sum := 0.0
	for _, idx := range indices {
		sum += values[idx]
	}
	return sum
}

func calculateMSE(predictions, actual []float64) float64 {
	sum := 0.0
	for i := range predictions {
		diff := predictions[i] - actual[i]
		sum += diff * diff
	}
	return sum / float64(len(predictions))
}

// SplitByDate splits data into train and test based on a cutoff date
func SplitByDate(data *StockData, cutoffDate string) (*StockData, *StockData, error) {
	cutoff, err := time.Parse("2006-01-02", cutoffDate)
	if err != nil {
		return nil, nil, fmt.Errorf("invalid date format: %v", err)
	}

	trainFeatures := make([][]float64, 0)
	trainTargets := make([]float64, 0)
	trainDates := make([]string, 0)

	testFeatures := make([][]float64, 0)
	testTargets := make([]float64, 0)
	testDates := make([]string, 0)

	for i := 0; i < len(data.Dates); i++ {
		dataDate, err := time.Parse("2006-01-02", data.Dates[i])
		if err != nil {
			continue
		}

		if dataDate.Before(cutoff) {
			trainFeatures = append(trainFeatures, data.Features[i])
			trainTargets = append(trainTargets, data.Target[i])
			trainDates = append(trainDates, data.Dates[i])
		} else {
			testFeatures = append(testFeatures, data.Features[i])
			testTargets = append(testTargets, data.Target[i])
			testDates = append(testDates, data.Dates[i])
		}
	}

	if len(trainDates) == 0 {
		return nil, nil, fmt.Errorf("no training data before cutoff date %s", cutoffDate)
	}
	if len(testDates) == 0 {
		return nil, nil, fmt.Errorf("no test data after cutoff date %s", cutoffDate)
	}

	trainData := &StockData{
		Features: trainFeatures,
		Target:   trainTargets,
		Dates:    trainDates,
	}
	testData := &StockData{
		Features: testFeatures,
		Target:   testTargets,
		Dates:    testDates,
	}

	return trainData, testData, nil
}

// Main function reading from historic_data directory
func main() {
	dataDir := "historic_data"

	fmt.Println("Loading CSV files from historic_data directory...")
	csvFiles, err := GetCSVFiles(dataDir)
	if err != nil {
		fmt.Printf("Error reading directory: %v\n", err)
		return
	}

	if len(csvFiles) == 0 {
		fmt.Printf("No CSV files found in %s directory\n", dataDir)
		return
	}

	fmt.Printf("Found %d CSV files\n\n", len(csvFiles))

	// Process each CSV file
	for _, csvFile := range csvFiles {
		symbol := filepath.Base(csvFile)
		fmt.Printf("\n========================================\n")
		fmt.Printf("Processing: %s\n", symbol)
		fmt.Printf("========================================\n")

		quotes, err := LoadStockDataFromCSV(csvFile)
		if err != nil {
			fmt.Printf("Error loading %s: %v\n", symbol, err)
			continue
		}

		fmt.Printf("Loaded %d days of data\n", len(quotes))

		// Sort quotes by date (oldest first)
		sort.Slice(quotes, func(i, j int) bool {
			return quotes[i].Date < quotes[j].Date
		})

		// Show date range
		if len(quotes) > 0 {
			firstDate := time.Unix(quotes[0].Date, 0).Format("2006-01-02")
			lastDate := time.Unix(quotes[len(quotes)-1].Date, 0).Format("2006-01-02")
			fmt.Printf("Date range: %s to %s\n", firstDate, lastDate)
		}

		// Prepare features and targets
		fmt.Println("Preparing features and targets...")
		stockData := PrepareStockData(quotes)
		if stockData == nil {
			fmt.Println("Not enough data to prepare features")
			continue
		}

		fmt.Printf("Prepared %d samples with %d features each\n", len(stockData.Target), len(stockData.Features[0]))

		// Configuration: Choose split method
		// Option 1: Use date-based split (uncomment to use)
		// useDateSplit := true
		// cutoffDate := "2025-01-01"

		// Option 2: Use percentage split (default)
		useDateSplit := false
		cutoffDate := "2024-11-01" // Only used if useDateSplit is true

		var trainData, testData *StockData

		if useDateSplit {
			// Date-based split
			var err error
			trainData, testData, err = SplitByDate(stockData, cutoffDate)
			if err != nil {
				fmt.Printf("Error splitting by date: %v\n", err)
				continue
			}
			fmt.Printf("\nUsing date-based split (cutoff: %s)\n", cutoffDate)
		} else {
			// Percentage-based split (80/20)
			trainSize := int(0.8 * float64(len(stockData.Target)))
			trainData = &StockData{
				Features: stockData.Features[:trainSize],
				Target:   stockData.Target[:trainSize],
				Dates:    stockData.Dates[:trainSize],
			}
			testData = &StockData{
				Features: stockData.Features[trainSize:],
				Target:   stockData.Target[trainSize:],
				Dates:    stockData.Dates[trainSize:],
			}
			fmt.Printf("\nUsing percentage-based split (80/20)\n")
		}

		// Show train/test split info
		fmt.Printf("Train set: %d samples (%s to %s)\n",
			len(trainData.Dates),
			trainData.Dates[0],
			trainData.Dates[len(trainData.Dates)-1])
		fmt.Printf("Test set: %d samples (%s to %s)\n",
			len(testData.Dates),
			testData.Dates[0],
			testData.Dates[len(testData.Dates)-1])

		// Create and train model with adjusted parameters
		model := NewXGBoost(100, 0.2, 200)
		model.Lambda = 0.01
		model.Gamma = 0.0

		fmt.Println("\n=== Training XGBoost Model ===")
		model.Train(trainData)

		// Evaluate on test data
		fmt.Println("\n=== Test Set Evaluation ===")
		testPredictions := model.Predict(testData.Features)
		testMSE := calculateMSE(testPredictions, testData.Target)
		fmt.Printf("Test MSE: %.20f\n", testMSE)

		// Show some predictions
		fmt.Println("\n=== Sample Predictions (Test Set) ===")
		fmt.Println("Date       | Predicted | Actual    | Error")
		fmt.Println("-----------|-----------|-----------|----------")
		for i := 0; i < min(10, len(testPredictions)); i++ {
			fmt.Printf("%s | %+.6f | %+.6f | %+.6f\n",
				testData.Dates[i],
				testPredictions[i],
				testData.Target[i],
				testPredictions[i]-testData.Target[i])
		}

		// Make prediction for the most recent data
		fmt.Println("\n=== Latest Prediction ===")
		latestFeatures := [][]float64{stockData.Features[len(stockData.Features)-1]}
		latestPrediction := model.Predict(latestFeatures)
		fmt.Printf("Predicted next day return: %+.4f%%\n", latestPrediction[0]*100)
	}
}

func calculateStdDev(values []float64) float64 {
	m := mean(values)
	variance := 0.0
	for _, v := range values {
		diff := v - m
		variance += diff * diff
	}
	variance /= float64(len(values))
	return math.Sqrt(variance)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
