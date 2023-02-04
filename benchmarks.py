import logging
import math
import functools
from time import perf_counter
from typing import Any, Callable

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    
    def bench_time(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = perf_counter()
            value = func(*args, **kwargs)
            end_time = perf_counter()
            run_time = end_time - start_time
            logging.info(f"Execution of {func.__name__} took {run_time:.2f} seconds.")
            return value

        return wrapper

    def bench_accuracy(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            value = func(*args, **kwargs)
            accuracy = 1.-abs(math.pi - value)/math.pi
            logging.info(f"Execution of {func.__name__} has an accuracy of {accuracy:.3f}.")
            return value

        return wrapper

if __name__ == "__main__":
    main()
    
