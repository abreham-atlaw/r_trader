import random
import time
import typing
from typing import List, Any
from abc import ABC

from copy import deepcopy
import datetime as dt
import attr

from stockfish import Stockfish
import chess
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import models

from lib.utils.logger import Logger
from lib.rl.agent import ActionChoiceAgent, ModelBasedAgent, MonteCarloAgent, ActionRecommendationAgent, \
	ActionRecommendationBalancerAgent, DNNTransitionAgent, Agent
from lib.rl.environment import ModelBasedState
from test.lib.rl.environment.environments.chess import ChessState
from lib.network.rest_interface import NetworkApiClient, Request
from lib.utils.stm import CueMemoryMatcher


class StockfishAgent(Agent):

	def __init__(self, elo=3000):
		super().__init__()
		self.__stockfish = Stockfish()
		self.__stockfish.set_elo_rating(elo)

	def __get_move(self, state: ChessState, san: str) -> chess.Move :
		board = deepcopy(state.get_board())
		board.push_san(san)
		return board.move_stack[-1]

	def _policy(self, state: ChessState) -> chess.Move:
		self.__stockfish.set_fen_position(state.get_board().fen())
		return self.__get_move(self.__stockfish.get_best_move())

class ChessActionChoiceAgent(ActionChoiceAgent, ABC):

	def _generate_actions(self, state: ChessState) -> List[chess.Move]:
		return list(state.get_board().legal_moves)


class ChessModelBasedAgent(ModelBasedAgent, ABC):

	class LiChessClient(NetworkApiClient):

		def __init__(self, token: str):
			super().__init__(
				url="https://explorer.lichess.ovh",
			)
			self.__token = token

		def execute(self, request: Request, headers: typing.Optional[typing.Dict] = None):
			if headers is None:
				headers = {}
			headers["Authorization"] = f"Bearer {self.__token}"
			return super().execute(request, headers)

	@attr.define
	class Stats:
		white: int
		draws: int
		black: int

		def total(self) -> int:
			return self.white + self.draws + self.black

	class StatsRequest(Request):

		def __init__(self, moves: typing.List[str]):
			super().__init__(
				url="masters",
				get_params={
					"play": ",".join(moves)
				},
				method=Request.Method.GET,
				output_class=ChessModelBasedAgent.Stats
			)

	LICHESS_TOKEN = "lip_b7sTGBkX2VOYg1YyrzfK"

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__lichess_client = ChessModelBasedAgent.LiChessClient(self.LICHESS_TOKEN)

	def __get_num_games(self, board: chess.Board) -> int:
		while True:
			try:
				return self.__lichess_client.execute(
					request=ChessModelBasedAgent.StatsRequest(
						moves=[str(move) for move in board.move_stack]
					)
				).total()
			except:
				time.sleep(2*60)
				return self.__get_num_games(board)

	def _get_expected_transition_probability_distribution(self, initial_states: List[ChessState], actions: List[chess.Move], final_states: List[ChessState]) -> List[float]:
		# return self.__get_num_games(final_state.get_board())/self.__get_num_games(initial_state.get_board())
		return [0.2 for _ in range(len(initial_states))]

	def _update_transition_probability(self, initial_states: ModelBasedState, action, final_state):
		pass

	def _get_expected_instant_reward(self, state: ChessState) -> float:
		return self._get_environment().get_reward(state)

	def _get_possible_states(self, state: ChessState, action: chess.Move) -> List[ModelBasedState]:
		mid_state = deepcopy(state)
		mid_state.get_board().push(action)

		if mid_state.get_board().is_game_over():
			return [mid_state]

		states = []

		for possible_action in mid_state.get_board().legal_moves:
			possible_state = deepcopy(mid_state)
			possible_state.get_board().push(possible_action)
			states.append(possible_state)
			del possible_state

		return states


class ChessStockfishModelBasedAgent(ModelBasedAgent, ABC):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__stockfish = Stockfish()
		self.__cache: typing.Dict[str, str] = {}

	def __get_single_expected_transition_probability(self, initial_state: ChessState, action: chess.Move, final_state: ChessState ) -> float:
		mid_state = deepcopy(final_state.get_board())
		mid_state.pop()
		fen = mid_state.fen()

		best_move = self.__cache.get(fen)
		if best_move is None:
			self.__stockfish.set_fen_position(mid_state.fen())
			best_move = self.__stockfish.get_best_move_time(time=1)
			self.__cache[fen] = best_move
		mid_state.push_san(best_move)
		return float(mid_state.move_stack[-1].uci() == final_state.get_board().move_stack[-1].uci())

	def _get_expected_transition_probability_distribution(self, initial_states: List[ChessState], actions: List[chess.Move], final_states: List[ChessState]) -> List[float]:

		return [
			self.__get_single_expected_transition_probability(initial_state, action, final_state)
			for initial_state, action, final_state in zip(initial_states, actions, final_states)
		]

	def _update_transition_probability(self, initial_states: ModelBasedState, action, final_state):
		pass


class ChessMonteCarloAgent(MonteCarloAgent, ABC):

	class ChessStateMemoryMatcher(CueMemoryMatcher):

		def is_match(self, cue: ChessState, memory: ChessState) -> bool:
			return cue.get_board() == memory.get_board() and cue.get_player_side() == memory.get_player_side()

	def __init__(self, *args, step_time=1*60, **kwargs):
		super().__init__(
			*args,
			short_term_memory=MonteCarloAgent.NodeShortTermMemory(
				100,
				MonteCarloAgent.NodeMemoryMatcher(
					ChessMonteCarloAgent.ChessStateMemoryMatcher()
				)
			),
			logical=True,
			**kwargs
		)
		self.__step_time = step_time

	def _init_resources(self) -> object:
		return dt.datetime.now()

	def _has_resource(self, resources: dt.datetime) -> bool:
		return (dt.datetime.now() - resources).total_seconds() < self.__step_time


class ChessDNNTransitionAgent(DNNTransitionAgent, ABC):

	TRANSITION_MODEL = None

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__model: models.Model = self._load_model()

	@staticmethod
	def _load_model():
		if (ChessDNNTransitionAgent.TRANSITION_MODEL == None):
			model = models.load_model("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/chessModel/out/model", compile=False)
			model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
			ChessDNNTransitionAgent.TRANSITION_MODEL = model
		return ChessDNNTransitionAgent.TRANSITION_MODEL

	def _get_transition_model(self) -> models.Model:
		return self.__model

	def __one_hot_encoding(self, classes: typing.List[typing.Any], class_: typing.Any) -> typing.List[float]:
		return [1 if class_ == c else 0 for c in classes]

	@staticmethod
	def __encode_board(board):
		encoding = np.zeros((14, 8, 8), dtype=int)
		for i in range(64):
			piece = board.piece_at(i)
			if piece is not None:
				encoding[piece.piece_type + int(piece.color) * 6][i // 8][i % 8] = 1
		encoding[13] = 1 if board.turn == chess.BLACK else 0
		return encoding.reshape(-1)

	def __prepare_single_dta_input(self, state: ChessState, action: chess.Move, final_state: ChessState) -> np.ndarray:
		mid_board = deepcopy(final_state.get_board())
		mid_board.pop()
		return np.array([self.__encode_board(mid_board)])

	def __prepare_single_dta_output(
			self,
			initial_state: ChessState,
			output: np.ndarray,
			final_state: ChessState
	) -> float:
		move = final_state.get_board().move_stack[-1]
		output = output.flatten()
		return output[move.from_square * 64 + move.to_square]

	def _prepare_dta_input(self, states: typing.List[ChessState], actions: List[typing.Any], final_states: typing.List[ChessState]) -> np.ndarray:
		return np.concatenate([
			self.__prepare_single_dta_input(state, action, final_state)
			for state, action, final_state in zip(states, actions, final_states)
		])

	def _prepare_dta_output(self, initial_states: typing.List[ModelBasedState], outputs: typing.List[np.ndarray],
							final_states: typing.List[ModelBasedState]) -> List[float]:
		return [
			self.__prepare_single_dta_output(initial_state, output, final_state)
			for initial_state, output, final_state in zip(initial_states, outputs, final_states)
		]

	def __encode_action(self, action):
		start_square = action.from_square
		end_square = action.to_square
		encoding = np.zeros(4096, dtype=int)
		encoding[start_square*64 + end_square] = 1
		return encoding

	def __prepare_single_dta_train_output(self, initial_state: ChessState, action: chess.Move, final_state: ChessState) -> np.ndarray:
		return self.__encode_action(final_state.get_board().move_stack[-1])

	def _prepare_dta_train_output(self, initial_states: typing.List[ChessState], actions: typing.List[chess.Move], final_states: typing.List[ChessState]) -> np.ndarray:
		return np.stack([
			self.__prepare_single_dta_train_output(initial_state, action, final_state)
			for initial_state, action, final_state in zip(initial_states, actions, final_states)
		])

	def _fit_model(self, X: np.ndarray, y: np.ndarray, fit_params: typing.Dict):
		print(f"[+]Fitting({self._get_environment().get_state().get_player_side()})...")
		self.__model.fit(X, y, epochs=5, verbose=1, batch_size=1)

	def _evaluate_model(self, X: np.ndarray, y: np.ndarray):
		self.__model.evaluate(X, [y[:, :64], y[:, 64:]])

	def _predict(self, model: models.Model, inputs: np.array) -> np.ndarray:
		out = model(inputs).numpy()
		return out


class ChessActionRecommenderAgent(ActionRecommendationAgent, ABC):

	PRE_TRAIN_MODEL_PATH = "chess_ara.h5"
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
	def __generate_pretrain_data(self) -> typing.Tuple[np.ndarray, typing.List[np.ndarray]]:  # TODO: MIGRATE
		boards = self.__generate_random_boards(self.PRE_TRAIN_BOARDS_NUM)
		X, y = [], None
		for board in boards:
			state = ChessState(board.turn, board)
			for j in range(self.PRE_TRAIN_MOVE_INDEXES_NUM):
				X.append(self._prepare_input(state, j).flatten())
				outs = self._prepare_train_outputs(state, random.choice(list(board.legal_moves)))
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
			for _ in range(2)
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

	def __prepare_input(self, state: ChessState, index: int) -> np.ndarray:
		return np.concatenate(
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
		)

	def _prepare_inputs(self, states: typing.List[typing.Any], indexes: typing.List[int]) -> np.ndarray:
		return np.array([
			self.__prepare_input(state, index)
			for state, index in zip(states, indexes)
		])

	@staticmethod
	def __get_class(classes: typing.List[object], values: typing.List[float]) -> typing.Any:
		return max(classes, key=lambda class_: values[classes.index(class_)])

	@staticmethod
	def __one_hot_encoding(classes: typing.List[typing.Any], class_: typing.Any) -> typing.List[float]:
		return [1 if class_ == c else 0 for c in classes]

	def __prepare_output(self, state: ChessState, output: typing.List[np.ndarray]) -> chess.Move:
		return chess.Move(
			from_square=self.__get_class(list(range(64)), list(output[0].flatten())),
			to_square=self.__get_class(list(range(64)), list(output[1].flatten()))
		)

	def _prepare_outputs(self, states: typing.List[typing.Any], outputs: typing.List[np.array]) -> typing.List[typing.Any]:

		sep_outputs = [
			[
				outputs[j][i]
				for j in range(len(outputs))
			]
			for i in range(len(outputs[0]))
		]
		return [
			self.__prepare_output(state, output)
			for state, output in zip(states, sep_outputs)
		]

	def __prepare_train_output(self, state: ChessState, action: chess.Move) -> typing.List[np.ndarray]:
		return [np.array(out) for out in [
			self.__one_hot_encoding(
				list(range(64)),
				action.from_square
			), self.__one_hot_encoding(
				list(range(64)),
				action.to_square
			)]]

	def _prepare_train_outputs(
			self,
			states: typing.List[typing.Any],
			actions: typing.List[typing.Any]
	) -> typing.List[np.ndarray]:
		outputs = None

		for state, action in zip(states, actions):
			outs = self.__prepare_train_output(state, action)
			if outputs is None:
				outputs = [[] for _ in outs]

			for i, out in enumerate(outs):
				outputs[i].append(out)

		return [np.array(out) for out in outputs]


class ChessActionRecommendationBalancerAgent(
	ActionRecommendationBalancerAgent,
	ChessActionRecommenderAgent,
	ChessActionChoiceAgent,
	ABC
):
	def _generate_static_actions(self, state: ChessState) -> typing.List[object]:
		return ChessActionChoiceAgent._generate_actions(self, state)
