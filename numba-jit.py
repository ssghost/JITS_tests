import numba
import random
import numpy as np
from .benchmarks import bench_time 
from .benchmarks import bench_accuracy

@bench_time
@bench_accuracy
@numba.jit
def monte_carlo_pi(ln: int):
    acc = 0
    for _ in range(ln):
        x = random.random()
        y = random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / ln

def main() -> None:
    for i in np.logspace(start = 5, stop = 9, num = 5):
        print(monte_carlo_pi(i))

if __name__ == "__main__":
    main()
    