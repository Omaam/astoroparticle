"""Power law observations.
"""
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from partical_xspec.python.observations.observation import Observation


class PowerlawGauss(Observation):
    """Power law model of xspec observed in Gauss distribution.
    """

    def __init__(self,
                 observation_size,
                 xspec_bijector=None,
                 energy_ranges_kev=None,
                 dtype=tf.float32):

        if xspec_bijector is None:
            xspec_bijector = tfb.Blockwise([
                tfb.Chain([tfb.Exp()]),
                tfb.Chain([tfb.Exp()])
            ])

        super(PowerlawPoisson, self).__init__(
            xspec_model_name="powerlaw",
            observation_size=observation_size,
            noise_distribution=tfd.Gauss,
            xspec_bijector=xspec_bijector,
            energy_ranges_kev=energy_ranges_kev,
            dtype=dtype
        )


class PowerlawPoisson(Observation):
    """Power law model of xspec observed in Poisson distribution.
    """

    def __init__(self,
                 observation_size,
                 xspec_bijector=None,
                 energy_ranges_kev=None,
                 dtype=tf.float32):

        if xspec_bijector is None:
            xspec_bijector = tfb.Blockwise([
                tfb.Chain([tfb.Exp()]),
                tfb.Chain([tfb.Exp()])
            ])

        super(PowerlawPoisson, self).__init__(
            xspec_model_name="powerlaw",
            observation_size=observation_size,
            noise_distribution=tfd.Poisson,
            xspec_bijector=xspec_bijector,
            energy_ranges_kev=energy_ranges_kev,
            dtype=dtype
        )
