import tensorflow as tf 
import numpy as np

@tf.function(jit_compile=True)
def monte_carlo_pi(ln: int):
    seedx = tf.random.set_seed(111111)
    seedy = tf.random.set_seed(222222) 
    random_x = tf.random.uniform(shape=(ln,), seed=seedx)
    random_y = tf.random.uniform(shape=(ln,), seed=seedy)
    return 4.0 * tf.sum(tf.where((random_x**2 + random_y**2 < 1.0), np.ones((ln,)), np.zeros((ln,))))/ ln

if __name__ == "__main__":
    print(monte_carlo_pi(1000000000))