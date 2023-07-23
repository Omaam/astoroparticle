"""Power law observations.
"""
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from partical_xspec.python.observations.observation import Observation


class PowerlawGauss(Observation):
    """Power law model of xspec observed in Gauss distribution.
    """

    def __init__(self, xspec_bijector=None, dtype=tf.float32):

        if xspec_bijector is None:
            xspec_bijector = tfb.Blockwise([
                tfb.Chain([tfb.Exp()]),
                tfb.Chain([tfb.Exp()])
            ])

        super(PowerlawPoisson, self).__init__(
            xspec_model_name="powerlaw",
            noise_distribution=tfd.Gauss,
            default_xspec_bijector=xspec_bijector,
            dtype=dtype
        )


class PowerlawPoisson(Observation):
    """Power law model of xspec observed in Poisson distribution.
    """

    def __init__(self, xspec_bijector=None, dtype=tf.float32):

        if xspec_bijector is None:
            xspec_bijector = tfb.Blockwise([
                tfb.Chain([tfb.Exp()]),
                tfb.Chain([tfb.Exp()])
            ])

        super(PowerlawPoisson, self).__init__(
            xspec_model_name="powerlaw",
            noise_distribution=tfd.Poisson,
            default_xspec_bijector=xspec_bijector,
            dtype=dtype
        )
