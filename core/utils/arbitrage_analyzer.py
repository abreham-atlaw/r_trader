import typing
from enum import Enum

import numpy as np


class ArbitrageAnalyzer:

	class Direction(Enum):
		UP = 0
		DOWN = 1

	def __init__(self):
		pass

	@staticmethod
	def __get_first_cross(
			sequence: np.ndarray,
			point: float,
			direction: 'ArbitrageAnalyzer.Direction'
	) -> typing.Optional[int]:

		coef = 1
		if direction == ArbitrageAnalyzer.Direction.DOWN:
			coef = -1

		ready = False
		for i, s in enumerate(sequence[:-1]):
			diff = (s - point) * coef
			if diff < 0:
				ready = True
				continue
			if ready and diff > 0:
				return i

	def __get_points_first_cross(
			self,
			sequence: np.ndarray,
			points: typing.List[float],
			directions: typing.List['ArbitrageAnalyzer.Direction']
	) -> typing.Tuple[typing.Optional[int], typing.Optional[int]]:

		if len(points) != len(directions):
			raise ValueError("Points and Directions Size Mismatch")

		crosses = list(filter(
			lambda t: t[1] is not None,
			[
				(i, self.__get_first_cross(sequence, point, direction))
				for i, (point, direction) in enumerate(zip(points, directions))
			]
		))

		if len(crosses) == 0:
			return None, None

		return min(crosses, key=lambda t: t[1])

	def __get_all_crosses(
			self,
			sequence: np.ndarray,
			point: float,
			direction: 'ArbitrageAnalyzer.Direction'
	) -> typing.List[int]:
		crosses = [0]

		while crosses[-1] is not None:
			crosses.append(self.__get_first_cross(sequence[crosses[-1] + 1:], point, direction))

		return crosses[1:-1]

	def __get_bounces_recursive(
			self,
			sequence: np.ndarray,
			points: typing.Tuple[float, float],
			next_cross: int
	) -> typing.List[int]:
		cross = self.__get_first_cross(
			sequence,
			points[next_cross],
			[ArbitrageAnalyzer.Direction.DOWN, ArbitrageAnalyzer.Direction.UP][next_cross]
		)
		if cross is None:
			return []
		return [cross] + self.__get_bounces_recursive(sequence[cross:], points, (next_cross + 1) % 2)

	def __get_bounces(self, sequence: np.ndarray, points: typing.Tuple[float, float]) -> typing.List[int]:

		points = sorted(points)
		i, first_cross = self.__get_points_first_cross(
			sequence,
			points,
			[ArbitrageAnalyzer.Direction.DOWN, ArbitrageAnalyzer.Direction.UP]
		)
		if first_cross is None:
			return []
		return [first_cross] + self.__get_bounces_recursive(sequence, points, next_cross=(i + 1) % 2)

	def __has_crossed(
			self,
			sequence: np.ndarray,
			points: typing.List[float],
			directions: typing.List['ArbitrageAnalyzer.Direction']
	) -> bool:

		if len(points) != len(directions):
			raise ValueError("Points and Directions Size Mismatch")
		return self.__get_points_first_cross(sequence, points, directions)[1] is not None

	def __has_bounced(self, sequence: np.ndarray, points: typing.Tuple[float, float], bounces: int) -> bool:
		return len(self.__get_bounces(sequence, points)) >= bounces

	def get_cross_probability(
			self,
			sequence: np.ndarray,
			zone_size: float,
			time_steps: int
	) -> float:

		sample_size = len(sequence) - int(time_steps) + 1

		return sum([
			int(self.__has_crossed(
				sequence=sequence[i: i+time_steps],
				points=np.array([zone_size, -zone_size])*sequence[i]/2 + sequence[i],
				directions=[ArbitrageAnalyzer.Direction.UP, ArbitrageAnalyzer.Direction.DOWN]
			))
			for i in range(sample_size)]
		)/sample_size

	def get_bounce_probability(
			self,
			sequence: np.ndarray,
			bounces: int,
			close_zone_size: float,
			bounce_zone_size: float,
			time_steps: int
	) -> float:
		sample_size = len(sequence) - time_steps + 1

		bounce_counts = 0
		for i in range(sample_size):
			sample_sequence = sequence[i: i+time_steps]
			_, close_point = self.__get_points_first_cross(
				sequence=sample_sequence,
				points=np.array([close_zone_size, -close_zone_size])*sample_sequence[0]/2 + sample_sequence[0],
				directions=[ArbitrageAnalyzer.Direction.UP, ArbitrageAnalyzer.Direction.DOWN]
			)

			if close_point is not None:
				sample_sequence = sample_sequence[:close_point+1]

			if self.__has_bounced(
				sequence=sample_sequence,
				points=np.array([bounce_zone_size, -bounce_zone_size])*sample_sequence[0]/2 + sample_sequence[0],
				bounces=bounces
			):
				bounce_counts += 1

		return bounce_counts/sample_size
