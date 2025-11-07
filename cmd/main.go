package main

import (
	"fmt"
	"github.com/ch55secake/hyperion/pkg/data"
	"github.com/ch55secake/hyperion/pkg/features"
	"github.com/ch55secake/hyperion/pkg/ml/randomforest"
)

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
		stockData, err := data.LoadDataFromCSV(resource)
		if err != nil {
			fmt.Println("Failed to load data from CSV")
		}

		fmt.Printf("Loaded %d days of data\n", len(stockData))

		strategy := data.NewTradingStrategy(stockData, 20000.0)

		fmt.Println("Extracting enhanced features...")
		extractedFeatures := features.ExtractFeatures(strategy)
		strategy.Features = extractedFeatures

		fmt.Printf("Generated %d feature vectors with 18 features each\n", len(strategy.Features))

		// Train with more trees and better parameters
		// 2000, 200, 18 - the penny stock merchant
		randomforest.Train(strategy, 2000, 200, 18) // 50 trees, max depth 8, min samples 15

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
