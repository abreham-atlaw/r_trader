import typing

from dataclasses import dataclass, field


@dataclass
class Method:
	name: str
	args: typing.Tuple = ()
	kwargs: typing.Dict = field(default_factory=lambda: {})
	loose: bool = False
	matrix: bool = False

	def call(self, instance):
		return getattr(instance, self.name)(*self.args, **self.kwargs)
