import numpy as np
import triton
import triton.language as tl
from .benchmarks import bench_time 
from .benchmarks import bench_accuracy

@bench_time
@bench_accuracy
@triton.jit
def monte_carlo_pi(ln: int):
    offset = tl.arange(ln)
    random_x = tl.rand(offset=offset)
    random_y = tl.rand(offset=offset)
    return 4.0 * tl.sum(tl.where((random_x**2 + random_y**2 < 1.0), tl.ones((ln,)), tl.zeros((ln,))))/ ln

def main() -> None:
    for i in np.logspace(start = 5, stop = 9, num = 5):
        print(monte_carlo_pi(i))

if __name__ == "__main__":
    main()