"""Xspec implemented observation function.
"""
import xspec

import tensorflow as tf
from tensorflow_probability import distributions as tfd


def get_observaton_function_xspec_poisson(
        model_str,
        num_particles,
        bijector=None,
        dtype=tf.float32):

    model = xspec.Model("powerlaw")
    x_param_names = []
    for comp_name in model.componentNames:
        x_param_names.extend(getattr(model, comp_name).parameterNames)
    num_params = len(x_param_names)

    def _observation_function(_, x):
        flux = []
        x_bijectored = bijector.forward(x)
        for i in range(num_particles):
            model = xspec.Model(model_str)
            for j in range(num_params):
                model(j+1).values = x_bijectored[i, j].numpy()
            flux.append(model.values(0))
        flux = tf.convert_to_tensor(flux, dtype=dtype)
        poisson = tfd.Independent(tfd.Poisson(flux),
                                  reinterpreted_batch_ndims=1)
        return poisson

    return _observation_function
