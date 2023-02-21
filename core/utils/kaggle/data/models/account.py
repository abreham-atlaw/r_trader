import typing
from abc import ABC, abstractmethod

from dataclasses import dataclass


@dataclass
class Account:
	username: str
	key: str
	