import numpy as np


def sigmoid(x):
	return 1/(1+np.e**(-x))


def moving_average(x: np.ndarray, window_size: int):
	data_len = len(x) - window_size + 1
	x_final = np.zeros(data_len)
	for i in range(data_len):
		x_final[i] = np.mean(x[i: i + window_size])
	return x_final
