import jax
import jax.numpy as jnp

@jax.jit
def monte_carlo_pi(random_x, random_y):
    return 4.0 * jnp.sum(jnp.where((random_x**2 + random_y**2 < 1.0), jnp.ones((ln,)), jnp.zeros((ln,))))/ ln

ln = 1000000000
keyx = jax.random.PRNGKey(111111)
keyy = jax.random.PRNGKey(222222) 
random_x = jax.random.uniform(keyx, shape=(ln,))
random_y = jax.random.uniform(keyy, shape=(ln,))

print(monte_carlo_pi(random_x, random_y))