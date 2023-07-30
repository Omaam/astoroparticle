# Partical Xspec Example

Example template of particle estimating using `astropf`.


## Template

We show the example where `observation` is powerlaw of xspec that
are observed with the Poisson noise and `transitions` is vector
autoregressive model that have the order of one.

```python
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

import astropf as px
from astropf import observations as pxo
from astropf import transitions as pxt


dtype = tf.float32
observed_values = tf.convert_to_tensor(
    np.loadtxt("path/to/observations.txt"), dtype=dtype)

observation = pxo.PowerlawPoisson()
transition = pxt.VectorAutoregressive(
    coefficients=[[[0.1, 0.0], [0.0, 0.0]]],
    noise_covariance=[[0.5, 0.0], [0.0, 0.2]],
    dtype=dtype)

pf = px.ParicleFilter(transition, observation,)

weited_particles = pf.sample(
    observed_values,
    init_latent_states=tfd.MultivariateNormalDiag(
        scale_diag=[0.5, 0.5]),
    num_particles=100)

filtered_particles = weited_particles.particles
smoothed_particles = weited_particles.smooth_lag_fix(fixed_lag=20)
```
In this way, we get `filtered_particles` and `smoothed_particles`.
