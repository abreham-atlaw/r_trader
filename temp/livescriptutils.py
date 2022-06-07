import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from lib.dnn.layers import *
from lib.network.oanda import Trader
from core import Config


def report_exception(ex):
	print(f"[-]Error", ex, ex.__class__)


def get_sequence(instrument, size, gran="M1"):
	while True:
		print(f"[+]Fetching {instrument}")
		try:
			candlesticks = trader.get_candlestick(instrument, count=size, to=datetime.now() - timedelta(minutes=1), granularity=gran)
			return np.array([float(point.mid["c"]) for point in candlesticks])
		except Exception as ex:
			report_exception(ex)


def create_trader():
	while True:
		try:
			return Trader(Config.OANDA_TOKEN, Config.OANDA_TRADING_ACCOUNT_ID)
		except Exception as ex:
			report_exception(ex)


def norm(X):
	return Norm()(X.reshape((1, -1))).numpy().reshape((-1,))

def sd(X):
	return np.sqrt(np.sum((X-np.average(X))**2)/len(X))


def moving_function(X, window_size, func):
	data_len = len(X) - window_size + 1
	X_final = np.zeros(data_len)
	for i in range(data_len):
		X_final[i] = func(X[i: i+window_size])
	return X_final


def ma(X, window):
	return moving_function(X, window, np.average)


def m_sd(X, window):
	return moving_function(X, window, sd)


def generate_core_data_for_instrument(instrument, size, seq_len=73, ma_window=10):
	total_length = size + seq_len + ma_window
	X = np.zeros((size, seq_len))
	y = np.zeros((size,))
	sequence = ma(get_sequence(total_length, instrument), ma_window)
	for i in range(size):
		X[i] = sequence[i:i+seq_len]
		if sequence[i+seq_len] > sequence[i+seq_len-1]:
			y[i] = 1
		else:
			y[i] = 0
	return X, y


def generate_core_data(instruments, size, seq_len=64, ma_window=10):
	X = np.zeros((0, seq_len))
	y = np.zeros(0)
	for instrument in instruments:
		ins_x, ins_y = generate_core_data_for_instrument(size, instrument, seq_len=seq_len, ma_window=ma_window)
		X = np.concatenate((X, ins_x))
		y = np.concatenate((y, ins_y))
	return X, y


def generate_delta_data_for_instrument(instrument, size, seq_len=73, ma_window=10):
	total_length = size + seq_len + ma_window
	X = np.zeros((size, seq_len+1))
	y = np.zeros((size))

	sequence = ma(get_sequence(instrument, total_length), ma_window)
	for i in range(size):
		direction = 0
		if sequence[i+seq_len] > sequence[i+seq_len-1]:
			direction = 1
		X[i] = np.append(
			sequence[i: i+seq_len],
			direction
		)
		y[i] = np.abs(sequence[i+seq_len] - sequence[i+seq_len - 1])
	return X, y


def generate_delta_data(instruments, size, seq_len=73, ma_window=10):
	X = np.zeros((0, seq_len))
	y = np.zeros(0)

	for instrument in instruments:
		ins_X, ins_y = generate_delta_data_for_instrument(instrument, size, seq_len=seq_len, ma_window=ma_window)
		X = np.concatenate((X, ins_X))
		y = np.concatenate((y, ins_y))
	return X, y


def delta(X):
	return X[1:] - X[:-1]


def stream_predictions(instrument, sleep_time=30):

	while True:
		sequence = get_sequence(instrument, core_model.input_shape[0])


trader = create_trader()
model_custom_objects = {layer.__name__: layer for layer in [Delta, Norm, MovingAverage, Percentage, UnNorm]}

instruments = trader.get_instruments()
delta_model = keras.models.load_model("res/delta_model_wrapped.h5", custom_objects=model_custom_objects)
core_model = keras.models.load_model("res/core_model_wrapped.h5", custom_objects=model_custom_objects)
