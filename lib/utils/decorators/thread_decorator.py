import threading


def thread_method(func):
	def wrapper(*args, **kwargs):
		thread = threading.Thread(target=func, args=args, kwargs=kwargs)
		thread.start()
		return thread

	return wrapper
