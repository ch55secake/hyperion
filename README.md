# hyperion-py

Responsible for training and testing the XBoost model built on stock data. 

## Endpoints 

The server package exposes two endpoints:

- `/train` - Trains the model provided in the body of the request
- `/predict/<ticker>` - Predicts the next 180 days of stock data for the given ticker
- `/trading_results/<ticker>` - Fetches the trading results from a given ticker

## Example requests 

Collection of example requests that can be fired at the service from the command line. 

### Train

Kicks off the training process for the ticker inside the request body. 

```bash
curl -vv -X POST http://localhost:8080/train -H 'Content-Type: application/json' -d @ticker_body.json
```

Ticker body example: 

```json 
{
  "ticker": "AAPL", 
  "period": "5y", 
  "interval": "1d"
}
```

### Predict 

Fetches the next 180 days of stock data for the given ticker, based on a previously trained model. 

```bash
curl -vv -X GET http://localhost:8080/predict/TSLA -H 'Content-Type: application/json'
```

### Trading results 

Fetches the trading results from a given ticker but requires that the model was previously trained. 

```bash 
curl -vv -X GET http://localhost:8080/trading_results/TSLA -H 'Content-Type: application/json'
```