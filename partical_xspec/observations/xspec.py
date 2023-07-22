"""Xspec implemented observation function.
"""
import xspec
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def get_observaton_function_xspec_poisson(
        model_str,
        xspec_param_size,
        num_particles,
        using_latent_indicies=None,
        bijector=None,
        dtype=tf.float32):

    model = xspec.Model(model_str)
    x_param_names = []
    for comp_name in model.componentNames:
        x_param_names.extend(getattr(model, comp_name).parameterNames)

    def _observation_function(_, x):
        if using_latent_indicies is not None:
            x = tf.gather(x, using_latent_indicies, axis=-1)
        if bijector is not None:
            x = bijector.forward(x)

        particle_flux = []
        for i in range(num_particles):
            model.setPars(*x[i].tolist())
            particle_flux.append(model.values(0))
        particle_flux = tf.convert_to_tensor(particle_flux, dtype=dtype)

        observation_dist = tfd.Independent(
            tfd.Poisson(particle_flux), reinterpreted_batch_ndims=1)
        return observation_dist

    return _observation_function
