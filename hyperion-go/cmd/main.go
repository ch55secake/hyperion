package main

import (
	"log"
    "time"
	"hyperion-go/pkg/client"
)

func main() {
	url := "http://localhost:8080"
	xgc := client.NewClient(url)

	pennyStocks := []string{
		"QD","DDL","RERE","IH","WDH","UXIN","TBLA","BZUN","DAO","KC",
		"SCWO","BDTX","CYH","BABB","TBTC","SVRA","VFF","NPWR","ME","GEVO",
	}

	for _, ticker := range pennyStocks {
		if err := xgc.Train(ticker); err != nil {
			log.Printf("failed to train %s: %v", ticker, err)
			continue
		}
        log.Printf("trained %s — waiting 20s before predict", ticker)
		time.Sleep(20 * time.Second)

		if err := xgc.Predict(ticker); err != nil {
			log.Printf("failed to predict %s: %v", ticker, err)
			continue
		}
		log.Printf("predicted %s", ticker)
	}
}
