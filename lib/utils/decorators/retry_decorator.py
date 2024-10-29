import time


def retry(exception_cls=Exception, patience=3, sleep_timer=0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < patience:
                try:
                    return func(*args, **kwargs)
                except exception_cls:
                    retries += 1
                    if retries < patience:
                        time.sleep(sleep_timer)
            raise exception_cls("Failed after {} retries".format(patience))
        return wrapper
    return decorator
