from lib.utils.timeout import timeout


def timeout_function(duration):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return timeout(lambda: func(*args, **kwargs), duration)
        return wrapper
    return decorator
