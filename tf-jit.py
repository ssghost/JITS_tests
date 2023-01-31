import tensorflow as tf 
import numpy as np
from .benchmarks import bench_time 
from .benchmarks import bench_accuracy

@bench_time
@bench_accuracy
@tf.function(jit_compile=True)
def monte_carlo_pi(ln: int):
    seedx = tf.random.set_seed(111111)
    seedy = tf.random.set_seed(222222) 
    random_x = tf.random.uniform(shape=(ln,), seed=seedx)
    random_y = tf.random.uniform(shape=(ln,), seed=seedy)
    return 4.0 * tf.sum(tf.where((random_x**2 + random_y**2 < 1.0), np.ones((ln,)), np.zeros((ln,))))/ ln

def main() -> None:
    for i in np.logspace(start = 5, stop = 9, num = 5):
        print(monte_carlo_pi(i))

if __name__ == "__main__":
    main()
    