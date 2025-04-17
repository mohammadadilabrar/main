import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import streamlit as st
from pyngrok import ngrok
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta



def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def preprocess_data(data, column='Close', lookback=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[column].values.reshape(-1, 1))

    training_data_len = int(np.ceil(len(scaled_data) * .95))

    train_data = scaled_data[0:int(training_data_len), :]
    x_train, y_train = [], []

    for i in range(lookback, len(train_data)):
        x_train.append(train_data[i-lookback:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler

def create_lstm_model(lookback):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def get_model_n_scaler(ticker: str, start_date: str, end_date: str, lookback: int = 60):
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    x_train, y_train, scaler = preprocess_data(stock_data, lookback=lookback)

    model = create_lstm_model(lookback)
    model.fit(x_train, y_train, batch_size=1, epochs=5)

    return model, scaler

def get_predicted_values(ticker : str, start_date : str, end_date : str, scaler, model, lookback : int = 60):
    live_data = yf.download(ticker, start=start_date, end=end_date)

    scaled_live_data = scaler.transform(live_data['Close'].values.reshape(-1, 1))

    x_live = []
    for i in range(lookback, len(scaled_live_data)):
        x_live.append(scaled_live_data[i-lookback:i, 0])

    x_live = np.array(x_live)
    x_live = np.reshape(x_live, (x_live.shape[0], x_live.shape[1], 1))

    predicted_prices = model.predict(x_live)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    return live_data, predicted_prices

def get_plot(live_data, lookback, predicted_prices, ticker):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=live_data.index, 
            y=live_data['Close']
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[str(x.date()) for x in live_data.index[lookback:].to_list()], 
            y=[x[0] for x in predicted_prices.tolist()], 
            mode='lines' 
        )
    )

    fig.update_layout(
        showlegend=False, 
        template="plotly_dark", 
        title={
            'text': f'{ticker} Stock Trend',
            'xanchor': 'auto',
            'yanchor': 'top',
        },
        xaxis_title="Date",
        yaxis_title="Predicted_Price",
        xaxis={'showgrid':False},
        # yaxis={'showgrid':False},
    )

    return fig
