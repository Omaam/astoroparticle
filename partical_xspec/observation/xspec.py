"""Xspec implemented observation function.
"""
import xspec
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


def get_observaton_function_xspec_poisson(
        model_str,
        num_particles,
        bijector=None,
        dtype=tf.float32):

    model = xspec.Model("powerlaw")
    x_param_names = []
    for comp_name in model.componentNames:
        x_param_names.extend(getattr(model, comp_name).parameterNames)
    model = xspec.Model(model_str)

    def _observation_function(_, x):
        flux = []
        x_bijectored = bijector.forward(x).numpy()
        for i in range(num_particles):
            # TODO: tolist() should not use.
            model.setPars(*x_bijectored[i].tolist())
            flux.append(model.values(0))
        flux = tf.convert_to_tensor(flux, dtype=dtype)
        poisson = tfd.Independent(tfd.Poisson(flux),
                                  reinterpreted_batch_ndims=1)
        return poisson

    return _observation_function
