import random
import typing


class ChoiceUtils:

	@staticmethod
	def __get_larger_list(a, b):
		if len(a) > len(b):
			return a
		return b

	@staticmethod
	def choice_discrete(a, b):
		return random.choice([a, b])

	@staticmethod
	def choice_continuous(
			a,
			b,
			round_mode=False,
			noise=None,
			min_value=None,
			max_value=None
	):

		if noise is not None:
			d = b - a
			a, b = a - d * noise, b + d * noise

		value = random.uniform(a, b)

		if min_value:
			value = max(min_value, value)

		if max_value:
			value = min(max_value, value)

		if round_mode:
			value = int(round(value))

		return value

	@staticmethod
	def list_select(
			a: typing.List,
			b: typing.List,
			discrete: bool = True,
			round_mode: bool = False,
			noise: float = None,
			size: int = None
	) -> typing.List:
		min_len = min(len(a), len(b))
		max_len = max(len(a), len(b))

		new_len = size
		if size is None:
			new_len = random.randint(min_len, max_len)

		mixed_values = []

		for i in range(min_len):

			if discrete:
				new_value = ChoiceUtils.choice_discrete(a[i], b[i])
			else:
				new_value = ChoiceUtils.choice_continuous(
					a[i],
					b[i],
					round_mode=round_mode,
					noise=noise
				)

			mixed_values.append(
				new_value
			)

		larger_list = ChoiceUtils.__get_larger_list(a, b)

		for i in range(min_len, new_len):
			mixed_values.append(
				larger_list[i]
			)

		return mixed_values

	@staticmethod
	def generate_list(
			a,
			b,
			size: typing.Union[typing.Tuple[int, int], int],
			discrete: bool = True,
			round_mode: bool = False,
	):
		if isinstance(size, tuple):
			size = ChoiceUtils.choice_continuous(
				*size,
				round_mode=True
			)

		return ChoiceUtils.list_select(
			[a for _ in range(size)],
			[b for _ in range(size)],
			discrete=discrete,
			round_mode=round_mode
		)
