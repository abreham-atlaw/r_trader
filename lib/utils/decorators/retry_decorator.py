import time
from lib.utils.logger import Logger


def retry(exception_cls=(Exception,), patience=3, sleep_timer=0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            exceptions = tuple(exception_cls) if isinstance(exception_cls, (list, tuple)) else (exception_cls,)
            while retries < patience:
                try:
                    return func(*args, **kwargs)
                except exceptions as ex:
                    retries += 1
                    if retries < patience:
                        Logger.warning(f"Exception Raised: {ex.__class__.__name__}, {ex}.\nRetrying... ({retries}/{patience})")
                        time.sleep(sleep_timer)
            raise exceptions[0]("Failed after {} retries".format(patience))
        return wrapper
    return decorator
