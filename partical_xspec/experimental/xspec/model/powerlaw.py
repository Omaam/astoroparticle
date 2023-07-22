"""Powerlaw model.
"""
from tensorflow_probability import bijectors as tfb

from partical_xspec.xspec.model.xspec_model import XspecModel


class Powerlaw(XspecModel):
    def __init__(self):

        bijector = tfb.Blockwise([
            tfb.Exp(),
            tfb.Exp()
        ])

        super(Powerlaw, self).__init__(
            model_name="powerlaw",
            param_size=2,
            bijector=bijector)
