import math
import logging
import functools
from typing import Any, Callable

def benchmark_accuracy(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        value = func(*args, **kwargs)
        accuracy = 1.-abs(math.pi - value)/math.pi
        logging.info(f"Execution of {func.__name__} has an accuracy of {accuracy:.2f}.")
        return value

    return wrapper