import numpy as np


def sigmoid(x):
	return 1/(1+np.e**(-x))


def delta(x):
	return x[1:] - x[:-1]


def moving_average(x: np.ndarray, window_size: int):
	data_len = len(x) - window_size + 1
	x_final = np.zeros(data_len)
	for i in range(data_len):
		x_final[i] = np.mean(x[i: i + window_size])
	return x_final


def kalman_filter(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:

	y = [x[0]]
	v = 0

	for i in range(1, len(x)):
		y.append(
			x[i] * alpha + (1 - alpha) * (y[i - 1] + v)
		)
		v = beta * (x[i] - y[i - 1]) + (1 - beta) * v

	return np.array(y)


def softmax(x):
	exp_x = np.exp(x - np.max(x))
	softmax_x = exp_x / np.sum(exp_x)
	return softmax_x
