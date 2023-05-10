import random
import typing
from typing import List
from abc import ABC

from copy import deepcopy
import datetime as dt

import chess
import numpy as np
from tensorflow.python.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import models

from lib.utils.logger import Logger
from lib.rl.agent import ActionChoiceAgent, ModelBasedAgent, MonteCarloAgent, ActionRecommendationAgent, ActionRecommendationBalancerAgent
from lib.rl.environment import ModelBasedState
from test.lib.rl.environment.environments.chess import ChessState


class ChessActionChoiceAgent(ActionChoiceAgent, ABC):

	def _generate_actions(self, state: ChessState) -> List[chess.Move]:
		return list(state.get_board().legal_moves)


class ChessModelBasedAgent(ModelBasedAgent, ABC):

	def _get_expected_transition_probability(self, initial_state: ModelBasedState, action, final_state) -> float:
		return 0.2

	def _update_transition_probability(self, initial_state: ModelBasedState, action, final_state):
		pass

	def _get_expected_instant_reward(self, state: ChessState) -> float:
		return self._get_environment().get_reward(state)

	def _get_possible_states(self, state: ChessState, action: chess.Move) -> List[ModelBasedState]:
		mid_state = deepcopy(state)
		mid_state.get_board().push(action)

		states = []

		for possible_action in mid_state.get_board().legal_moves:
			possible_state = deepcopy(mid_state)
			possible_state.get_board().push(possible_action)
			states.append(possible_state)
			del possible_state

		return states


class ChessMonteCarloAgent(MonteCarloAgent, ABC):

	def _init_resources(self) -> object:
		return dt.datetime.now()

	def _has_resource(self, resources: dt.datetime) -> bool:
		return (dt.datetime.now() - resources).total_seconds() < 1*60


class ChessActionRecommenderAgent(ActionRecommendationAgent, ABC):

	PRE_TRAIN_MODEL_PATH = "./chess_ara.h5"
	PRE_TRAIN_BOARDS_NUM = 10
	PRE_TRAIN_MOVE_INDEXES_NUM = 10
	PRE_TRAIN_MAX_MOVE_DEPTH = 10

	def __generate_random_board(self) -> typing.Optional[chess.Board]:
		board = chess.Board()
		num_moves = random.randint(0, self.PRE_TRAIN_MAX_MOVE_DEPTH)
		for _ in range(num_moves):
			possible_moves = list(board.legal_moves)
			if len(possible_moves) == 0:
				return None
			board.push(random.choice(possible_moves))
		return board

	def __generate_random_boards(self, size) -> typing.List[chess.Board]:
		boards = []
		while len(boards) < size:
			board = self.__generate_random_board()
			if board is not None and len(list(board.legal_moves)) > 0:
				boards.append(board)
		return boards

	@Logger.logged_method
	def __generate_pretrain_data(self) -> typing.Tuple[np.ndarray, typing.List[np.ndarray]]:
		boards = self.__generate_random_boards(self.PRE_TRAIN_BOARDS_NUM)
		X, y = [], None
		for board in boards:
			state = ChessState(board.turn, board)
			for j in range(self.PRE_TRAIN_MOVE_INDEXES_NUM):
				X.append(self._prepare_input(state, j).flatten())
				outs = self._prepare_train_output(state, random.choice(list(board.legal_moves)))
				if y is None:
					y = [[] for _ in outs]
				for i, out in enumerate(outs):
					y[i].append(out)

		return np.array(X), [np.array(o) for o in y]

	@Logger.logged_method
	def __pretrain_model(self, model: Model):
		X, y = self.__generate_pretrain_data()
		model.fit(X, y, batch_size=8, epochs=10)

	def __generate_new_model(self) -> Model:
		inputs = layers.Input((65,))
		norm = layers.BatchNormalization()(inputs)
		dense = norm
		for layer_size in [64, 64, 64]:
			dense = layers.Dense(layer_size, activation="relu")(dense)
		outs = [
			layers.Dense(64, activation="softmax")(dense)
			for i in range(2)
		]
		model = Model(inputs=inputs, outputs=outs)
		model.compile(optimizer="adam", loss=["categorical_crossentropy" for _ in range(2)])
		model.summary()

		self.__pretrain_model(model)
		return model

	def _init_ara_model(self) -> Model:
		try:
			return models.load_model(self.PRE_TRAIN_MODEL_PATH)
		except Exception as ex:
			model = self.__generate_new_model()
			model.save(self.PRE_TRAIN_MODEL_PATH)
			return model

	def _prepare_input(self, state: ChessState, index: int) -> np.ndarray:
		return np.expand_dims(
			np.concatenate(
				[
					self.__one_hot_encoding(list(range(-6, 7)), piece)
					for piece in [
						state.get_board().piece_at(i).piece_type * 2*(int(state.get_board().piece_at(i).color) - 0.5)
						if state.get_board().piece_at(i) is not None
						else 0
						for i in range(64)
					]
				] + [[index, int(state.get_board().turn)]],
				axis=0
				),
			axis=0
		)

	@staticmethod
	def __get_class(classes: typing.List[object], values: typing.List[float] ) -> typing.Any:
		return max(classes, key=lambda class_: values[classes.index(class_)])

	@staticmethod
	def __one_hot_encoding(classes: typing.List[typing.Any], class_: typing.Any) -> typing.List[float]:
		return [1 if class_ == c else 0 for c in classes]

	def _prepare_output(self, state: ChessState, output: typing.List[np.ndarray]) -> chess.Move:
		return chess.Move(
			from_square=self.__get_class(list(range(64)), list(output[0].flatten())),
			to_square=self.__get_class(list(range(64)), list(output[1].flatten()))
		)

	def _prepare_train_output(self, state: ChessState, action: chess.Move) -> typing.List[np.ndarray]:
		return [np.array(out) for out in [
			self.__one_hot_encoding(
				list(range(64)),
				action.from_square
			), self.__one_hot_encoding(
				list(range(64)),
				action.to_square
			)]]


class ChessActionRecommendationBalancerAgent(
	ActionRecommendationBalancerAgent,
	ChessActionRecommenderAgent,
	ChessActionChoiceAgent,
	ABC
):
	def _generate_static_actions(self, state: ChessState) -> typing.List[object]:
		return ChessActionChoiceAgent._generate_actions(self, state)
