import time
import typing

import chess
import chess.svg

from threading import Thread
from copy import deepcopy

from lib.rl.environment import Environment, ModelBasedState
from lib.rl.agent.agent import Agent


class ChessState(ModelBasedState):

	def __init__(self, side, board: chess.Board):
		self.__depth = 0
		self.__board = board
		self.__side = side
		self.__current_player = chess.WHITE

	def set_depth(self, depth: int):
		self.__depth = depth

	def get_depth(self) -> int:
		return self.__depth

	def get_board(self) -> chess.Board:
		return self.__board

	def get_current_player(self) -> int:
		return self.__board.turn

	def get_player_side(self) -> int:
		return self.__side

	def switch_current_player(self):
		self.__current_player = not self.__current_player


class ChessEnvironment(Environment):

	def __init__(self, *args, state=None, **kwargs):
		super().__init__(*args, **kwargs)
		if state is None:
			state = ChessState(chess.WHITE, chess.Board())
		self.__state = state

	def __get_game_end_reward(self, state: ChessState) -> float:
		if state.get_board().is_checkmate():
			if state.get_current_player() == state.get_player_side():
				return 1
			else:
				return -1
		if state.get_board().is_stalemate():
			return 0
		return 0

	def __get_pieces_value(self, board, color) -> int:
		return sum([
			board.piece_at(i).piece_type
			for i in range(64)
			if
			board.piece_at(i) is not None and board.piece_at(i).color == color
		])

	def get_reward(self, state: ChessState = None) -> float:
		if state is None:
			state = self.get_state()

		game_end_reward = self.__get_game_end_reward(state)
		pieces_reward = self.__get_pieces_value(state.get_board(), state.get_player_side()) - self.__get_pieces_value(state.get_board(), not state.get_player_side())
		return 10 * game_end_reward + pieces_reward

	def perform_action(self, action: chess.Move):
		self.get_state().get_board().push(action)
		self.get_state().switch_current_player()
		print(
			f"[+](Player: {['Black', 'White'][int(self.get_state().get_player_side())]}): Waiting for {['Black', 'White'][int(self.get_state().get_current_player())]}...")
		while self.get_state().get_current_player() != self.get_state().get_player_side() and not self.is_episode_over():
			time.sleep(1)
		print(f"[+](Player: {['Black', 'White'][int(self.get_state().get_player_side())]}): Continuing...")

	def __print_board(self, board: chess.Board):
		for i in range(7, -1, -1):
			for j in range(8):
				if j == 0:
					print("|", end="")
				piece = board.piece_at(i * 8 + j)
				char = f"{' ': ^15}"
				if piece is not None:
					char = f"{piece.unicode_symbol(): ^14}"
				print(char, end="|")
			print("\n" + "".join(["_" for i in range(127)]))

	def render(self):
		self.__print_board(self.get_state().get_board())

	def update_ui(self):
		self.render()

	def check_is_running(self) -> bool:
		return not (self.is_episode_over())

	def is_action_valid(self, action: chess.Move, state: ChessState) -> bool:
		return state.get_board().is_legal(action)

	def get_state(self) -> ChessState:
		return self.__state

	def is_episode_over(self, state=None) -> bool:
		if state is None:
			state = self.get_state()
		return state.get_board().is_checkmate() or state.get_board().is_stalemate()


class ChessGame:

	class PlayerThread(Thread):

		def __init__(self, agent: Agent, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.__agent = agent

		def run(self) -> None:
			self.__agent.perform_episode()

	def __init__(self, player0: Agent, player1: Agent, board: typing.Optional[chess.Board] = None):
		self.__players = player0, player1
		if board is None:
			board = chess.Board()
		self.__players[0].set_environment(
			ChessEnvironment(state=ChessState(chess.WHITE, board))
		)

		self.__players[1].set_environment(
			ChessEnvironment(state=ChessState(chess.BLACK, board))
		)

	def start(self):

		threads = []
		for i in range(2):
			player_thread = ChessGame.PlayerThread(self.__players[i])
			player_thread.start()
			threads.append(player_thread)

		for thread in threads:
			thread.join()
