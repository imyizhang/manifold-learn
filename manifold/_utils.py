import functools


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
