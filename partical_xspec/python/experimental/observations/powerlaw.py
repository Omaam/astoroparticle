"""Power law observations.
"""
from partical_xspec.python.experimental.observations.observation import Observation
from partical_xspec.python.experimental.xspec.model.powerlaw import Powerlaw


class PowerlawPoisson(Observation):

    def __init__(self):
        super(PowerlawPoisson, self).__init__(
            xspec_model_name="powerlaw",
            noise_model_name="poisson")
