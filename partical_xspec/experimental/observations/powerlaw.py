"""Power law observations.
"""
from partical_xspec.experimental.observations.observation import Observation
from partical_xspec.experimental.xspec.model.powerlaw import Powerlaw


class PowerlawPoisson(Observation):

    def __init__(self):
        super(PowerlawPoisson, self).__init__(
            xspec_model_name="powerlaw",
            noise_model_name="poisson")
