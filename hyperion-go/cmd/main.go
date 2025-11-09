package main

import (
	"hyperion-go/pkg/client"
	"log"
)

func main() {
	url := "http://localhost:8080"
	xgc := client.NewClient(url)

	pennyStocks := []string{
		"QD", "DDL", "RERE", "IH", "WDH", "UXIN", "TBLA", "BZUN", "DAO", "KC",
		"SCWO", "BDTX", "CYH", "BABB", "TBTC", "SVRA", "VFF", "NPWR", "ME", "GEVO",
	}

	for _, ticker := range pennyStocks {
		if err := xgc.Train(ticker); err != nil {
			log.Printf("failed to train %s: %v", ticker, err)
			continue
		}

		log.Printf("Getting the trading results for %s\n", ticker)

		if err := xgc.TradingResults(ticker); err != nil {
			log.Printf("failed to train %s: %v", ticker, err)
			continue
		}
	}
}
