package data

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"
	"time"
)

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
