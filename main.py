import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
import logging


TICKER = 'GOOG' 
START_DATE = '2023-01-01'
END_DATE = '2024-01-01'
TIME_STEP = 60 
EPOCHS = 50  
BATCH_SIZE = 32
N_SPLITS = 5 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError(f"No data fetched for ticker {ticker}. Please check the ticker symbol and try again.")
        return stock_data
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        exit()

def preprocess_data(stock_data):
    stock_data['Lag_1'] = stock_data['Close'].shift(1) 
    stock_data['MA_7'] = stock_data['Close'].rolling(window=7).mean()  
    stock_data['MA_30'] = stock_data['Close'].rolling(window=30).mean()
    stock_data['Return'] = stock_data['Close'].pct_change() 
    stock_data.dropna(inplace=True) 
    return stock_data


def evaluate_model(model, X_test, Y_test, model_name):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(Y_test, predictions)
    mse = mean_squared_error(Y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, predictions)
    mape = mean_absolute_percentage_error(Y_test, predictions)
    logging.info(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, MAPE: {mape:.2f}")
    return predictions, mae, rmse, r2, mape


def tune_random_forest(X_train, Y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    logging.info(f"Best parameters for Random Forest: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_xgboost(X_train, Y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    model = XGBRegressor(random_state=42)
    grid_search = RandomizedSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, n_iter=10)
    grid_search.fit(X_train, Y_train)
    logging.info(f"Best parameters for XGBoost: {grid_search.best_params_}")
    return grid_search.best_estimator_


def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])  # Use past 'time_step' days as input
        Y.append(data[i + time_step, 0])  # Use the next day's value as target
    return np.array(X), np.array(Y)

def plot_results(Y_test, predictions_lr, predictions_rf, predictions_xgb, predictions_ensemble, fold):
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test, label='Actual Prices', color='blue', linewidth=2)
    plt.plot(predictions_lr, label='Linear Regression', color='orange', linestyle='--')
    plt.plot(predictions_rf, label='Random Forest', color='green', linestyle='-.')
    plt.plot(predictions_xgb, label='XGBoost', color='red', linestyle=':')
    plt.plot(predictions_ensemble, label='Ensemble', color='purple', linestyle='-')
    plt.title(f'{TICKER} Stock Price Prediction (Fold {fold + 1})')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

def main():
   
    stock_data = fetch_stock_data(TICKER, START_DATE, END_DATE)
    logging.info(stock_data.head())
    stock_data = preprocess_data(stock_data)
    logging.info(stock_data.shape)

  
    X = stock_data[['Lag_1', 'MA_7', 'MA_30', 'Return']].values
    Y = stock_data['Close'].values

   
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = []

    plot_figures = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        logging.info(f"Fold {fold + 1}: Train indices: {train_index}, Test indices: {test_index}")
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Linear Regression
        model_lr = LinearRegression()
        model_lr.fit(X_train, Y_train)
        predictions_lr, mae_lr, rmse_lr, r2_lr, mape_lr = evaluate_model(model_lr, X_test, Y_test, "Linear Regression")

       
        model_rf = tune_random_forest(X_train, Y_train)
        predictions_rf, mae_rf, rmse_rf, r2_rf, mape_rf = evaluate_model(model_rf, X_test, Y_test, "Random Forest")

       
        model_xgb = tune_xgboost(X_train, Y_train)
        predictions_xgb, mae_xgb, rmse_xgb, r2_xgb, mape_xgb = evaluate_model(model_xgb, X_test, Y_test, "XGBoost")

        # Ensemble model (Voting Regressor)
        ensemble_model = VotingRegressor([
            ('lr', model_lr),
            ('rf', model_rf),
            ('xgb', model_xgb)
        ])
        ensemble_model.fit(X_train, Y_train)
        predictions_ensemble, mae_ensemble, rmse_ensemble, r2_ensemble, mape_ensemble = evaluate_model(ensemble_model, X_test, Y_test, "Ensemble")

      
        results.append({
            'fold': fold + 1,
            'Linear Regression': (mae_lr, rmse_lr, r2_lr, mape_lr),
            'Random Forest': (mae_rf, rmse_rf, r2_rf, mape_rf),
            'XGBoost': (mae_xgb, rmse_xgb, r2_xgb, mape_xgb),
            'Ensemble': (mae_ensemble, rmse_ensemble, r2_ensemble, mape_ensemble)
        })

        
        plot_results(Y_test, predictions_lr, predictions_rf, predictions_xgb, predictions_ensemble, fold)
        plot_figures.append(plt.gcf()) 

    # LSTM Model
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Close']].values)

    X_lstm, Y_lstm = create_dataset(scaled_data, TIME_STEP)
    X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

    model_lstm = create_lstm_model((X_lstm.shape[1], 1))
    model_lstm.fit(X_lstm, Y_lstm, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    predictions_lstm = model_lstm.predict(X_lstm)
    predictions_lstm = scaler.inverse_transform(predictions_lstm)
    Y_lstm_original = scaler.inverse_transform(Y_lstm.reshape(-1, 1))

    # Plot LSTM results
    plt.figure(figsize=(10, 6))
    plt.plot(Y_lstm_original, label='Actual Prices', color='blue', linewidth=2)
    plt.plot(predictions_lstm, label='LSTM Predictions', color='red', linestyle='--')
    plt.title(f'{TICKER} Stock Price Prediction (LSTM Model)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plot_figures.append(plt.gcf())  
   

    plt.show()

if __name__ == "__main__":
    main()
