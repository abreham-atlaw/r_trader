from dataclasses import dataclass

from lib.rl.agent.mca.node import Node


@dataclass
class NodeMemory:
	node: Node
