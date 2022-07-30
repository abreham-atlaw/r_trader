from typing import *
from abc import ABC, abstractmethod

import unittest
from dataclasses import dataclass
import random
import math

from lib.ga import GeneticAlgorithm, Species


class GeneticAlgorithmTest(unittest.TestCase):

	class Positions:

		@dataclass
		class Position:
			name: str
			position: Tuple[float, float]

		GK = Position("GK", (2, 0))
		LB = Position("LB", (0, 5))
		CB = Position("CB", (2, 4))
		RB = Position("RB", (4, 5))
		DMF = Position("DMF", (2, 7.5))
		LMF = Position("LMF", (0, 10))
		CMF = Position("CMF", (2, 10))
		RMF = Position("RMF", (4, 10))
		AMF = Position("AMF", (2, 12.5))
		LW = Position("LW", (0, 14))
		RW = Position("RW", (4, 14))
		SS = Position("SS", (2, 14))
		CF = Position("CF", (2, 15))

	@dataclass
	class Player:

		name: str
		overall: float
		preferred_position: 'GeneticAlgorithmTest.Positions.Position'

		def __str__(self):
			return self.name

	class FootballFormation(Species):

		def __init__(self, players, formation):
			self.__players = players
			self.__formation = formation

		def get_players(self):
			return self.__players

		def get_formation(self):
			return self.__formation

		def mutate(self, *args, **kwargs):
			for i in range(1):
				x, y = random.randint(0, 10), random.randint(0, 10)
				self.__players[x], self.__players[y] = self.__players[y], self.__players[x]

		def __generate_offspring_players(self, spouse):
			players = []
			for i in range(len(self.get_players())):
				candidates = [self.get_players()[i], spouse.get_players()[i]]
				player = None
				while player is None or player in players:
					if len(candidates) == 0:
						candidates = [
							random.choice(
								[
									player
									for player in self.get_players()
									if player not in players
								] +
								[
									player
									for player in spouse.get_players()
									if player not in players
								]
							)
							for _ in range(1)
						]
					player = random.choice(candidates)
					candidates.remove(player)
				players.append(player)
			return players

		def __generate_offspring(self, spouse: 'GeneticAlgorithmTest.FootballFormation'):

			players = self.__generate_offspring_players(spouse)
			formation = random.choice([spouse.get_formation(), self.get_formation()])
			return GeneticAlgorithmTest.FootballFormation(players, formation)

		def reproduce(self, spouse: 'GeneticAlgorithmTest.FootballFormation', preferred_offsprings: int) -> List['GeneticAlgorithmTest.FootballFormation']:
			return [
				self.__generate_offspring(spouse)
				for i in range(preferred_offsprings)
			]

		def __str__(self):
			return "-".join([f"{str(player)}({position.name})" for player, position in zip(self.get_players(), self.get_formation())])

	class FootballFormationOptimizer(GeneticAlgorithm):

		def __init__(self, squad: 'List[GeneticAlgorithmTest.Player]', possible_formations, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.__squad = squad
			self.__formations = possible_formations


		def __generate_random_formation(self):
			players = self.__squad.copy()
			random.shuffle(players)
			formation = random.choice(self.__formations)
			return GeneticAlgorithmTest.FootballFormation(players, formation)

		def _generate_initial_generation(self) -> List[Species]:
			return [
				self.__generate_random_formation()
				for _ in range(20)
			]

		def __eucleadian_distance(self, p0, p1) -> float:
			return math.sqrt(
				sum([
					(p0[0] - p1[0])**2 + (p0[1] - p1[1])**2
				])
			)

		def _evaluate_species(self, species: 'GeneticAlgorithmTest.FootballFormation') -> float:
			value = 0
			for player, position in zip(species.get_players(), species.get_formation()):
				distance = self.__eucleadian_distance(player.preferred_position.position, position.position)
				if distance == 0:
					distance = 0.00001
				player_value = player.overall * 1/distance
				value += player_value
			return value

	ARSENAL_SQUAD = [
				Player(
					"Leno",
					84,
					Positions.GK,
				),
				Player(
					"Monreal",
					80,
					Positions.LB,
				),
				Player(
					"Gabriel",
					80,
					Positions.CB,
				),
				Player(
					"Mustafi",
					83,
					Positions.CB,
				),
				Player(
					"Bellerin",
					83,
					Positions.RB,
				),
				Player(
					"Xhaka",
					83,
					Positions.DMF,
				),
				Player(
					"Ramsay",
					84,
					Positions.CMF,
				),
				Player(
					"Ozil",
					88,
					Positions.AMF,
				),
				Player(
					"Aubameyang",
					86,
					Positions.LW,
				),
				Player(
					"Pepe",
					81,
					Positions.RW,
				),
				Player(
					"Lacazette",
					84,
					Positions.CF,
				),

			]
	ARGENTINA_SQUAD = [
		Player(
			"Messi",
			95,
			Positions.RW
		),
		Player(
			"Dybala",
			89,
			Positions.SS
		),
		Player(
			"Martinez",
			78,
			Positions.GK,
		),
		Player(
			"Di Maria",
			87,
			Positions.RW
		),
		Player(
			"Marscherano",
			81,
			Positions.CB
		),
		Player(
			"Otamendi",
			84,
			Positions.CB
		),
		Player(
			"Higuain",
			85,
			Positions.CF
		),
		Player(
			"Gaitan",
			84,
			Positions.LMF
		),
		Player(
			"Gomez",
			83,
			Positions.CMF
		),
		Player(
			"Molina",
			81,
			Positions.RB
		),
		Player(
			"De Paul",
			80,
			Positions.CB
		)

	]

	FourThreeThree = [Positions.GK, Positions.LB, Positions.CB, Positions.CB, Positions.RB, Positions.DMF, Positions.CMF, Positions.AMF, Positions.LW, Positions.RW, Positions.CF]
	FourFourTwo = [Positions.GK, Positions.LB, Positions.CB, Positions.CB, Positions.RB, Positions.LMF, Positions.CMF, Positions.CMF, Positions.RMF, Positions.SS, Positions.CF]
	ThreeFourThree = [Positions.GK, Positions.CB, Positions.CB, Positions.CB, Positions.DMF, Positions.CMF, Positions.CMF, Positions.AMF, Positions.LW, Positions.RW, Positions.CF]

	def test_functionality(self):
		optimizer = GeneticAlgorithmTest.FootballFormationOptimizer(
			GeneticAlgorithmTest.ARGENTINA_SQUAD,
			[
				GeneticAlgorithmTest.FourFourTwo,
				GeneticAlgorithmTest.FourThreeThree,
				GeneticAlgorithmTest.ThreeFourThree
			],
			generation_growth_factor=1
		)
		final_generation = optimizer.start(200)

