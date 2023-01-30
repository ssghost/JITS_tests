import numba
import random
from . import benchmark_time, benchmark_accuracy

@benchmark_time
@benchmark_accuracy
@numba.jit
def monte_carlo_pi(ln: int):
    acc = 0
    for _ in range(ln):
        x = random.random()
        y = random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / ln

if __name__ == "__main__":
    print(monte_carlo_pi(1000000000))