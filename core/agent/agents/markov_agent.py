from lib.rl.agent import MarkovAgent
from .dnn_transition_agent import TraderDNNTransitionAgent


class TraderMarkovAgent(MarkovAgent, TraderDNNTransitionAgent):

	def __init__(self, *args, **kwargs):
		super(TraderMarkovAgent, self).__init__(*args, **kwargs)
