package client

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

type PredictionRequest struct {
	Ticker string `json:"ticker"`
	Date   string `json:"date"`
}

type TrainRequest struct {
	Ticker string `json:"ticker"`
}

type XGClient struct {
	client http.Client
	url    string
}

func NewClient(url string) *XGClient {
	return &XGClient{
		client: http.Client{},
		url:    url,
	}
}

func (c *XGClient) Train(ticker string) error {
	body := TrainRequest{
		Ticker: ticker,
	}

	bodyBytes, err := json.Marshal(body)
	req, err := http.NewRequest(http.MethodPost, c.url, bytes.NewBuffer(bodyBytes))
	if err != nil {
		panic(err)
	}

	req.Header.Set("Content-Type", "application/json")
	response, err := c.client.Do(req)
	if err != nil {
		return errors.New(fmt.Sprintf("Error sending request: %s", err))
	}

	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return errors.New(fmt.Sprintf("Error reading response: %s", err))
	}
	fmt.Println(string(responseBody))
	return nil
}

func (c *XGClient) Predict(ticker, date string) error {
	body := PredictionRequest{
		Ticker: ticker,
		Date:   date,
	}

	bodyBytes, err := json.Marshal(body)
	req, err := http.NewRequest(http.MethodGet, c.url, bytes.NewBuffer(bodyBytes))
	if err != nil {
		panic(err)
	}

	req.Header.Set("Content-Type", "application/json")
	response, err := c.client.Do(req)
	if err != nil {
		return errors.New(fmt.Sprintf("Error sending request: %s", err))
	}

	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return errors.New(fmt.Sprintf("Error reading response: %s", err))
	}
	fmt.Println(string(responseBody))
	return nil

}
