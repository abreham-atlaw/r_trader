import signal

from lib.utils.logger import Logger


class TimeoutException(Exception):
    pass


def handle_timeout(*args, **kwargs):
    raise TimeoutException()


def timeout(func, duration):

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(duration)

    try:
        result = func()
    except TimeoutException:
        Logger.info(f"Function {func.__name__} timed out after {duration} seconds.")
        result = None
    return result
