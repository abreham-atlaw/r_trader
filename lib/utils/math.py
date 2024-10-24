import numpy as np


def sigmoid(x):
	return 1/(1+np.e**(-x))


def moving_average(x: np.ndarray, window_size: int):
	data_len = len(x) - window_size + 1
	x_final = np.zeros(data_len)
	for i in range(data_len):
		x_final[i] = np.mean(x[i: i + window_size])
	return x_final


def softmax(x):
	exp_x = np.exp(x - np.max(x))
	softmax_x = exp_x / np.sum(exp_x)
	return softmax_x
