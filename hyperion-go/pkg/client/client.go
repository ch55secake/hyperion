package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

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
	const period = "1y"
	intervals := []string{"1h", "1d"}

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

		// 2) Write to a file like ./train_bodies/AAPL_1d.json
		dir := "train_bodies"
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create dir %s: %w", dir, err)
		}
		filename := filepath.Join(dir, fmt.Sprintf("%s_%s.json", ticker, interval))
		if err := os.WriteFile(filename, bodyBytes, 0o644); err != nil {
			return fmt.Errorf("write %s: %w", filename, err)
		}

		// 3) Read back from the file (parity with `-d @file`) and POST
		fileBytes, err := os.ReadFile(filename)
		if err != nil {
			return fmt.Errorf("read %s: %w", filename, err)
		}

		req, err := http.NewRequest(
			http.MethodPost,
			fmt.Sprintf("%s/train", c.url),
			bytes.NewReader(fileBytes),
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

		fmt.Printf("Trained %s interval=%s period=%s OK (body: %s)\n", ticker, interval, period, filename)
	}

	return nil
}

func (c *XGClient) Predict(ticker string) error {
	body := PredictionRequest{Ticker: ticker}
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal predict %s: %w", ticker, err)
	}

	req, err := http.NewRequest(http.MethodPost, fmt.Sprintf("%s/predict", c.url), bytes.NewBuffer(bodyBytes))
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
