import triton
import triton.language as tl
from . import benchmark_time, benchmark_accuracy

@benchmark_time
@benchmark_accuracy
@triton.jit
def monte_carlo_pi(ln: int):
    offset = tl.arange(ln)
    random_x = tl.rand(offset=offset)
    random_y = tl.rand(offset=offset)
    return 4.0 * tl.sum(tl.where((random_x**2 + random_y**2 < 1.0), tl.ones((ln,)), tl.zeros((ln,))))/ ln

if __name__ == "__main__":
    print(monte_carlo_pi(1000000000))