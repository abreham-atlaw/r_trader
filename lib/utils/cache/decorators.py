from datetime import datetime

from .cache import Cache


class CacheDecorators:

	@staticmethod
	def __get_method_cache(instance, method, size):
		attribute_name = f"{method.__name__}__cache"
		if not hasattr(instance, attribute_name):
			setattr(instance, attribute_name, Cache(cache_size=size))
		return getattr(instance, attribute_name)

	@staticmethod
	def __get_cache_keys(args, kwargs, timeout=None) -> str:
		if timeout is None:
			time_key = "null",
		else:
			now = datetime.now()
			time_key = str(now.replace(minute=(now.minute//timeout)*timeout, second=0, microsecond=0).timestamp())
			print(f"Time Key: {time_key}(Timeout={timeout}, now={now})")
		return str({
			"args": [str(a) for a in args],
			"kwargs": {
				key: str(kwargs.get(key))
				for key in kwargs
			},
			"time": time_key
		})

	@staticmethod
	def cached_method(timeout=None, size=1000):
		def decorator(func):
			def wrapper(self, *args, **kwargs):
				cache = CacheDecorators.__get_method_cache(self, func, size=size)
				return cache.cached_or_execute(
					CacheDecorators.__get_cache_keys(args, kwargs, timeout=timeout),
					func=lambda: func(self, *args, **kwargs)
				)
			return wrapper
		return decorator
