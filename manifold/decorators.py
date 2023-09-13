import functools
import time


def alias(**aliases):
    """Decorator for aliasing keyword arguments in a function."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for param_alias, param_name in aliases.items():
                if param_alias in kwargs:
                    kwargs[param_name] = kwargs[param_alias]
                    del kwargs[param_alias]
            return func(*args, **kwargs)

        return wrapper

    return decorator


def timeit(func):
    """Decorator for measuring execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        since = time.time()
        result = func(*args, **kwargs)
        print(
            f"'{func.__name__}' executed, wall time: {time.time() - since:.4f} s"
        )
        return result

    return wrapper


def vtimeit(func):
    """Decorator for measuring execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        verbose = kwargs.get("verbose", False)
        if verbose:
            since = time.time()
        result = func(*args, **kwargs)
        if verbose:
            print(
                f"'{func.__name__}' executed, wall time: {time.time() - since:.4f} s"
            )
        return result

    return wrapper
