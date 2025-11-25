# Stock Predictions
Testing multiple models — from a simple LSTM model to a stacked ensemble transformer-based method — to predict changes in stock prices.

## Project Structure
1. **Basic LSTM Model**
2. **Stacked Ensemble Model**

---

# Basic LSTM Method

The `pricePredictor` class downloads historical stock data, normalizes it, trains several LSTM models with different hyperparameters, selects the best model based on RMSE, and produces a next-day predicted closing price.

### The class automatically:
1. Downloads historical data for a given ticker  
2. Normalizes closing prices  
3. Splits data into train/test sets  
4. Trains multiple LSTM models using the given hyperparameters  
5. Selects the best-performing model (lowest RMSE)  
6. Predicts the next day’s closing price  

---

## Requirements

### Dependencies  
Install all required packages:

```bash
pip install numpy pandas tensorflow scikit-learn yfinance
```
### Hyperparameters defined by the user
- `pUnits` → list of LSTM units to test  
- `pActivations` → list of activation functions to test  
- `pLearningRates` → list of learning rates to test  

The model with the lowest RMSE on the test set is selected as the final model.

## Example Usage

```
predictor = pricePredictor(
  pTicker='AAPL',
  pUnits=[16, 32],
  pActivations=['relu', 'tanh'],
  pLearningRates=[0.001, 0.005]
)

print("Next-day prediction:", predictor.get_prediction())
print("Best RMSE:", predictor.get_rmse())
best_model = predictor.get_best_model()
```


   
