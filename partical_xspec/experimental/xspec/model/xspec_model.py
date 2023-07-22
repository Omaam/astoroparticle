"""Xspec model module.
"""
import xspec


class XspecModel:
    def __init__(self, model_name, param_size, bijector):
        self.model_name = model_name
        self.param_size = param_size
        self.bijector = bijector

    @property
    def xspec_model(self):
        return xspec.Model(self.model_name)

    @property
    def default_bijector(self):
        return self._default_bijector

    def _default_bijector(self):
        raise NotImplementedError()
