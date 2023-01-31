import jax
import jax.numpy as jnp
import numpy as np
from .benchmarks import bench_time 
from .benchmarks import bench_accuracy

@bench_time
@bench_accuracy
@jax.jit
def monte_carlo_pi(ln: int):
    keyx = jax.random.PRNGKey(111111)
    keyy = jax.random.PRNGKey(222222) 
    random_x = jax.random.uniform(keyx, shape=(ln,))
    random_y = jax.random.uniform(keyy, shape=(ln,))
    return 4.0 * jnp.sum(jnp.where((random_x**2 + random_y**2 < 1.0), jnp.ones((ln,)), jnp.zeros((ln,))))/ ln


def main() -> None:
    for i in np.logspace(start = 5, stop = 9, num = 5):
        print(monte_carlo_pi(i))

if __name__ == "__main__":
    main()