from typing import *

import pandas as pd
import numpy as np


class MockDataGenerator:

	def __init__(
			self,
			size: int,
			*args,
			p=2, s=3, a=2, b=1, f=100,
			**kwargs):
		super().__init__(*args, **kwargs)
		self.__size = size
		self.__p = p
		self.__s = s
		self.__a = a
		self.__b = b
		self.__f = f

	def __eval(self, x) -> np.ndarray:
		return self.__b + np.sum([
			np.sin(
				(x*3.14*((i-1)**self.__p)/self.__size) + (i**self.__s)
			)/(i**self.__a)
			for i in range(1, self.__f+1)
		], axis=0)

	def start(self, out_file: str, instrument: Tuple[str, str]):
		df = pd.DataFrame(columns=["base_currency", "quote_currency", "c"])
		df["c"] = self.__eval(np.arange(self.__size))
		df["base_currency"], df["quote_currency"] = instrument
		print("Head\n\n", df.head())
		df.to_csv(out_file)
