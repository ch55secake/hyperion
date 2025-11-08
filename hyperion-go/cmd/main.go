package main

import (
	"fmt"
	"log"
	"time"

	"hyperion-go/pkg/client"

	"github.com/go-co-op/gocron"
)

func main() {
	url := "http://localhost:8080"
	xgc := client.NewClient(url)

	tickersToSchedule := []string{"AAPL", "MSFT", "NFLX"}

	s := gocron.NewScheduler(time.UTC)

	for _, ticker := range tickersToSchedule {
		_, err := s.Every(30).Seconds().Do(func() error {
			err := xgc.Train(ticker)
			if err != nil {
				return fmt.Errorf("failed to fetch tickers: %v", err)
			}
			return nil
		})
		if err != nil {
			log.Fatalf("failed to schedule job: %v", err)
		}
	}

	s.StartBlocking()
}
