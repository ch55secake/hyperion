package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type DataInfo struct {
	Period       string `json:"period"`
	TestSamples  int    `json:"test_samples"`
	TotalSamples int    `json:"total_samples"`
	TrainSamples int    `json:"train_samples"`
}

type ModelPerformance struct {
	TestMae  float64 `json:"test_mae"`
	TestR2   float64 `json:"test_r2"`
	TestRmse float64 `json:"test_rmse"`
}

type BestStrategy struct {
	FinalValue     float64 `json:"final_value"`
	Name           string  `json:"name"`
	NumberOfTrades int     `json:"number_of_trades"`
	TotalReturn    float64 `json:"total_return"`
}

type TradingSimulation struct {
	BestStrategy     BestStrategy `json:"best_strategy"`
	BuyAndHoldReturn interface{}  `json:"buy_and_hold_return"`
	InitialCapital   float64      `json:"initial_capital"`
	Strategies       struct{}
}

type Strategies struct {
	AdaptiveThreshold struct {
		FinalValue     float64 `json:"final_value"`
		NumberOfTrades int     `json:"number_of_trades"`
		TotalReturn    float64 `json:"total_return"`
	} `json:"adaptive_threshold"`
	Directional struct {
		FinalValue     float64 `json:"final_value"`
		NumberOfTrades int     `json:"number_of_trades"`
		TotalReturn    float64 `json:"total_return"`
	} `json:"directional"`
	HoldDays struct {
		FinalValue     float64 `json:"final_value"`
		NumberOfTrades int     `json:"number_of_trades"`
		TotalReturn    float64 `json:"total_return"`
	} `json:"hold_days"`
}

type TradingResults struct {
	PredictionResults struct {
		DataInfo          DataInfo          `json:"data_info"`
		ModelPerformance  ModelPerformance  `json:"model_performance"`
		Ticker            string            `json:"ticker"`
		TradingSimulation TradingSimulation `json:"trading_simulation"`
	} `json:"prediction_results"`
	Status string `json:"status"`
}

type PredictionRequest struct {
	Ticker string `json:"ticker"`
}

type TrainRequest struct {
	Ticker   string `json:"ticker"`
	Interval string `json:"interval"`
	Period   string `json:"period"`
}

type XGClient struct {
	url    string
	client *http.Client
}

func NewClient(url string) *XGClient {
	return &XGClient{
		url:    url,
		client: &http.Client{},
	}
}

// Train writes a JSON file per (ticker, interval) and posts it to /train.
func (c *XGClient) Train(ticker string) error {
	const period = "10y"
	intervals := []string{"1d"}

	for _, interval := range intervals {
		payload := TrainRequest{
			Ticker:   ticker,
			Interval: interval,
			Period:   period,
		}

		// 1) Marshal JSON
		bodyBytes, err := json.MarshalIndent(payload, "", "  ")
		if err != nil {
			return fmt.Errorf("marshal %s (%s): %w", ticker, interval, err)
		}

		req, err := http.NewRequest(
			http.MethodPost,
			fmt.Sprintf("%s/train", c.url),
			bytes.NewReader(bodyBytes),
		)
		if err != nil {
			return fmt.Errorf("build request %s (%s): %w", ticker, interval, err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.client.Do(req)
		if err != nil {
			return fmt.Errorf("send %s (%s): %w", ticker, interval, err)
		}
		defer resp.Body.Close()

		respBody, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("status %d for %s (%s): %s", resp.StatusCode, ticker, interval, string(respBody))
		}

		fmt.Printf("Trained %s interval=%s period=%s OK\n", ticker, interval, period)
	}

	return nil
}

func (c *XGClient) TradingResults(ticker string) error {
	req, err := http.NewRequest(http.MethodGet, fmt.Sprintf("%s/trading-results/%s", c.url, ticker), nil)
	if err != nil {
		return fmt.Errorf("build predict request %s: %w", ticker, err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("send predict %s: %w", ticker, err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("predict status %d for %s: %s", resp.StatusCode, ticker, string(respBody))
	}

	var tr TradingResults
	err = json.Unmarshal(respBody, &tr)
	if err != nil {
		return err
	}

	marshalledTradingResults, err := json.MarshalIndent(tr, "", "  ")
	if err != nil {
		fmt.Println("error whilst doing marshal indent:", err)
	}

	fmt.Printf(string(marshalledTradingResults))
	fmt.Printf("\n")
	fmt.Printf(strings.Repeat("=", 60))
	fmt.Printf("\n")
	if tr.PredictionResults.ModelPerformance.TestR2 > 0.012 {
		fmt.Printf("The test MAE is %f\n", tr.PredictionResults.ModelPerformance.TestMae)
		fmt.Printf("The test R2 is %f\n", tr.PredictionResults.ModelPerformance.TestR2)
		fmt.Printf("The test RMSE is %f\n", tr.PredictionResults.ModelPerformance.TestRmse)
	}
	fmt.Printf("Just chalk it and try again.\n")
	fmt.Printf(strings.Repeat("=", 60))

	return nil
}

func (c *XGClient) Predict(ticker string) error {
	req, err := http.NewRequest(http.MethodGet, fmt.Sprintf("%s/predict/%s", c.url, ticker), nil)
	if err != nil {
		return fmt.Errorf("build predict request %s: %w", ticker, err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("send predict %s: %w", ticker, err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("predict status %d for %s: %s", resp.StatusCode, ticker, string(respBody))
	}

	fmt.Printf("Prediction for %s: %s\n", ticker, string(respBody))
	return nil
}
