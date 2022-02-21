from typing import *
from dataclasses import dataclass

import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.layers import InputLayer, Dense, Softmax
from tensorflow.keras.activations import relu
from tensorflow.keras.losses import CategoricalCrossentropy

import os
from copy import deepcopy
from random import shuffle, randint, choice


from lib.rl.agent import DNNTransitionAgent, MarkovAgent
from lib.rl.environment import Environment
from lib.utils.logger import Logger


class DNNAgentTest(unittest.TestCase):

	class BlackJackEnvironment(Environment):

		class OppositionPlayer:

			def __init__(self):
				pass

			def get_action(self, state) -> int:
				if state.opposition_state.get_sum() > 15:
					return DNNAgentTest.BlackJackEnvironment.Action.DONE
				return DNNAgentTest.BlackJackEnvironment.Action.HIT

		class Rewards:

			SUCCESS = 10
			LOSS = -10
			DRAW = 0
			TIME_PENALTY = -1

		@dataclass
		class Card:

			class Suit:

				CLUBS = 0
				DIAMONDS = 1
				HEARTS = 2
				SPADES = 3

			suit: int
			value: int

			def __str__(self):
				return f"Value: {self.value} Suit {self.suit}"

		class Deck:

			def __init__(self):
				self.cards = self.generate_cards()

			def generate_cards(self):
				return [
					DNNAgentTest.BlackJackEnvironment.Card(suit, value)
					for suit in [
						DNNAgentTest.BlackJackEnvironment.Card.Suit.CLUBS,
						DNNAgentTest.BlackJackEnvironment.Card.Suit.DIAMONDS,
						DNNAgentTest.BlackJackEnvironment.Card.Suit.HEARTS,
						DNNAgentTest.BlackJackEnvironment.Card.Suit.SPADES,
					]
					for value in range(1, 14)
				]

			def shuffle(self):
				shuffle(self.cards)

			def pop_card(self):
				return self.cards.pop()

			def get_size(self):
				return len(self.cards)

			def reset(self):
				self.cards = self.generate_cards()

		@dataclass
		class State:

			@dataclass
			class PlayerState:
				cards: List
				done: bool

				def get_sum(self):
					return sum([card.value for card in self.cards])

				def __str__(self):
					return f"Cards: {self.cards}\nDone: {self.done}"

			player_state: PlayerState
			opposition_state: PlayerState

			def is_all_done(self):
				return self.player_state.done and self.opposition_state.done

			def __str__(self):
				return f"Player0:\n{self.player_state}\n\nPlayer1:\n{self.opposition_state}"

		class Action:
			HIT = 0
			DONE = 1

		def __init__(self):
			super().__init__(episodic=True)
			self.deck = DNNAgentTest.BlackJackEnvironment.Deck()
			self.opposition_player = DNNAgentTest.BlackJackEnvironment.OppositionPlayer()
			self.state: Union[DNNAgentTest.BlackJackEnvironment.State, None] = None
			self.current_player = 1

		def get_state(self, transparent=False) -> State:
			if self.state is None:
				raise Exception("State not Initialized")
			if self.state.is_all_done() or transparent:
				return self.state
			state = deepcopy(self.state)
			state.opposition_state.cards = [None]*len(state.opposition_state.cards)
			return state

		@Logger.logged_method
		def get_winner(self, state: State) -> Union[int, None]:

			player_cards_sum = state.player_state.get_sum()
			opposition_cards_sum = state.opposition_state.get_sum()
			if (player_cards_sum > 21 and opposition_cards_sum > 21) or player_cards_sum == opposition_cards_sum:
				return None
			if player_cards_sum > 21 or 21 >= opposition_cards_sum > player_cards_sum:
				return 1 # Player 1 WINS
			if opposition_cards_sum > 21 or 21 >= player_cards_sum > opposition_cards_sum:
				return 0 # Player 0 WINS
			raise Exception(f"Invalid State {state}")

		@Logger.logged_method
		def get_reward(self, state: State=None) -> float:
			if state is None:
				state = self.get_state()
			if not state.is_all_done():
				return DNNAgentTest.BlackJackEnvironment.Rewards.TIME_PENALTY
			winner = self.get_winner(state)
			if winner is None:
				return DNNAgentTest.BlackJackEnvironment.Rewards.DRAW
			if winner == 0:
				return DNNAgentTest.BlackJackEnvironment.Rewards.SUCCESS
			if winner == 1:
				return DNNAgentTest.BlackJackEnvironment.Rewards.LOSS

			raise Exception(f"Invalid Winner {winner}")

		def swap_turn(self):
			self.current_player = (self.current_player + 1)%2

		def perform_opposition_action(self):
			action = self.opposition_player.get_action(self.get_state(transparent=True))
			self.perform_action(action)

		def perform_action(self, action: int):
			if self.current_player == 0:
				active_player_state = self.get_state(transparent=True).player_state
			else:
				active_player_state = self.get_state(transparent=True).opposition_state

			if action == DNNAgentTest.BlackJackEnvironment.Action.HIT:
				if self.deck.get_size() == 0:
					self.deck = DNNAgentTest.BlackJackEnvironment.Deck()
					self.deck.shuffle()
				active_player_state.cards.append(self.deck.pop_card())
			elif action == DNNAgentTest.BlackJackEnvironment.Action.DONE:
				active_player_state.done = True
				self.swap_turn()

		def render(self):
			print(self.state)

		def update_ui(self):
			os.system("clear")
			self.render()

		def check_is_running(self) -> bool:
			return not self.is_episode_over()

		@Logger.logged_method
		def get_valid_actions(self, state=None) -> List:
			return [
				DNNAgentTest.BlackJackEnvironment.Action.HIT,
				DNNAgentTest.BlackJackEnvironment.Action.DONE
			]

		@Logger.logged_method
		def is_episode_over(self, state=None) -> bool:
			if state is None:
				state = self.get_state()
			return state.player_state.done and state.opposition_state.done

		def _initialize(self):
			self.deck.shuffle()
			self.state = DNNAgentTest.BlackJackEnvironment.State(
				DNNAgentTest.BlackJackEnvironment.State.PlayerState(
					cards=[self.deck.pop_card() for i in range(2)],
					done=False
				),
				DNNAgentTest.BlackJackEnvironment.State.PlayerState(
					cards=[self.deck.pop_card() for i in range(2)],
					done=False
				)
			)
			super()._initialize()

		def start(self):
			super().start()
			while not self.get_state(transparent=True).is_all_done() and self.current_player == 1:
					self.perform_opposition_action()

		def reset(self):
			self.deck.reset()
			self._initialize()

	class BlackJackAgent(MarkovAgent, DNNTransitionAgent):

		class BlackJackTransitionModel(keras.Model):

			def __init__(self, *args, **kwargs):
				super().__init__(*args, **kwargs)
				self.hit_model, self.done_model = self.create_models()

			def create_models(self) -> Tuple[keras.Model, keras.Model]:
				hit_model = keras.Sequential()
				hit_model.add(InputLayer((1,)))
				hit_model.add(Dense(4, activation=relu))
				hit_model.add(Dense(2, activation=relu))
				hit_model.add(Softmax())

				done_model = keras.Sequential()
				done_model.add(InputLayer((2,)))
				done_model.add(Dense(8, activation=relu))
				done_model.add(Dense(3, activation=relu))
				done_model.add(Softmax())

				return hit_model, done_model

			def compile(self, *args, **kwargs):
				self.hit_model.compile(*args, **kwargs)
				self.done_model.compile(*args, **kwargs)
				return super().compile(*args, **kwargs)

			@Logger.logged_method
			def call(self, inputs, training=None, mask=None):
				outputs = []
				for input_ in inputs:
					if input_[2] == 0:
						output = tf.concat(
							(self.hit_model(tf.reshape(input_[0], (1, -1))), tf.zeros((1,3))),
							axis=1
						)[0]
					else:
						output = tf.concat(
							(tf.zeros((1, 2)), self.done_model(tf.reshape(input_[:2], (1, -1)))),
							axis=1
						)[0]
					outputs.append(output)

				return outputs

		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.set_transition_model(self.create_model())

		def create_model(self) -> keras.Model:
			model = keras.Sequential()
			model.add(InputLayer((3,)))
			model.add(Dense(8, activation=relu))
			model.add(Dense(16, activation=relu))
			model.add(Dense(5, activation=relu))
			model.add(Softmax())

			model.compile(
				optimizer='adam',
				loss=CategoricalCrossentropy()
			)

			return model

		@Logger.logged_method
		def _get_expected_instant_reward(self, state) -> float:
			return self._get_environment().get_reward(state)

		def generate_cards_within_range(self, range_: Tuple[int, int], cards_len: int) -> List:
			def gen_cards(values):
				return [
					DNNAgentTest.BlackJackEnvironment.Card(
						suit=choice([
							DNNAgentTest.BlackJackEnvironment.Card.Suit.SPADES,
							DNNAgentTest.BlackJackEnvironment.Card.Suit.HEARTS,
							DNNAgentTest.BlackJackEnvironment.Card.Suit.DIAMONDS,
							DNNAgentTest.BlackJackEnvironment.Card.Suit.CLUBS
						]),
						value = value
					)
					for value in values
				]
			if cards_len == 1:
				return gen_cards([randint(*range_)])
			simpler_range = (max(range_[0]-((cards_len-1)*13), 1), min(range_[1]-(cards_len-1), 13))
			new_card = self.generate_cards_within_range(simpler_range, 1)[0]
			return [new_card] + self.generate_cards_within_range((max(range_[0]-new_card.value, 1), min(range_[1]-new_card.value, (cards_len-1)*13)), cards_len-1)

		@Logger.logged_method
		def _get_possible_states(self, state, action) -> List[object]:
			player_cards_sum = state.player_state.get_sum()
			possible_player_cards = []
			possible_opposition_cards = []
			if action == DNNAgentTest.BlackJackEnvironment.Action.HIT:
				possible_player_cards = []
				if player_cards_sum < 21:
					possible_player_cards.append(
						state.player_state.cards + self.generate_cards_within_range((1, 21 - player_cards_sum), 1)
					)

				if player_cards_sum + 13 > 21:
					possible_player_cards.append(
						state.player_state.cards + self.generate_cards_within_range((21 - player_cards_sum + 1, 13), 1)
					)
				possible_opposition_cards = [state.opposition_state.cards]

			if action == DNNAgentTest.BlackJackEnvironment.Action.DONE:
				possible_player_cards = [state.player_state.cards]

				opp_cards_len = len(state.opposition_state.cards)
				possible_opposition_cards = [self.generate_cards_within_range((22, 13*opp_cards_len), opp_cards_len)]
				if player_cards_sum < 21:
					possible_opposition_cards.append(
						self.generate_cards_within_range((1, player_cards_sum), opp_cards_len)
					)
					possible_opposition_cards.append(
						self.generate_cards_within_range((player_cards_sum+1, 21), opp_cards_len)
					)
				else:
					possible_opposition_cards.append(
						self.generate_cards_within_range((1, 21), opp_cards_len)
					)

			possible_states = []
			for player_cards in possible_player_cards:
				for opposition_cards in possible_opposition_cards:
					new_state = deepcopy(state)
					new_state.player_state.cards = player_cards
					new_state.opposition_state.cards = opposition_cards
					if action == DNNAgentTest.BlackJackEnvironment.Action.DONE:
						new_state.player_state.done = True
					possible_states.append(new_state)

			return possible_states

		def get_final_state_index(self, initial_state, final_state) -> int:

			my_sum = final_state.player_state.get_sum()

			if len(initial_state.player_state.cards) < len(final_state.player_state.cards): # Action = HIT
				if my_sum <= 21:
					return 0
				if my_sum > 21:
					return 1

			opp_sum = final_state.opposition_state.get_sum()

			if opp_sum < my_sum <= 21:
				return 1

			if my_sum <= opp_sum <= 21 or opp_sum <= 21 < my_sum:
				return 2

			if opp_sum > 21:
				return 3

			raise Exception(f"Invalid Final-State {final_state}")

		@Logger.logged_method
		def _state_action_to_model_input(self, state, action, final_state) -> np.ndarray:
			return np.array([
				state.player_state.get_sum(),
				len(state.opposition_state.cards),
				action,
			])

		@Logger.logged_method
		def _prediction_to_transition_probability(self, initial_state, output: np.ndarray, final_state) -> float:
			return float(output[0][self.get_final_state_index(initial_state, final_state)])

		@Logger.logged_method
		def _get_train_output(self, initial_state, action, final_state) -> np.ndarray:
			output = np.zeros((5,))
			output[self.get_final_state_index(initial_state, final_state)] = 1
			return output

	def setUp(self) -> None:
		self.environment = DNNAgentTest.BlackJackEnvironment()
		self.environment.start()
		self.agent = DNNAgentTest.BlackJackAgent(depth=2, explore_exploit_tradeoff=1, update_batch_size=10)
		self.agent.set_environment(self.environment)

	def test(self):
		while True:
			self.environment = DNNAgentTest.BlackJackEnvironment()
			self.environment.start()
			self.agent.set_environment(self.environment)
			self.agent.perform_episode()


