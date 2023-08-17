"""Xspec handling module.
"""
import numpy as np
try:
    import xspec
except ModuleNotFoundError:
    print("can't find 'xspec' module, and ignore importing this.")


class XspecModel:
    def __init__(self, model_name, params=None):
        # Default xspec chatter level is 10.
        # 0 is completely quiet.
        XspecModel.set_chatter(0)

        self._model = xspec.Model(model_name)
        self.spectrum_loaded = False

    def compute_flux(self):
        if self.spectrum_loaded:
            return np.array(self._model.values(1))
        else:
            return np.array(self._model.values(0))

    def load_responses(self,
                       dammy_pha,
                       rmf="USE_DEFAULT",
                       arf="USE_DEFAULT"):
        xspec.Spectrum(dammy_pha, respFile=rmf, arfFile=arf)
        self.spectrum_loaded = True

    def set_parameters(self, params):
        param_dict = {}
        for comp_name in self._model.componentNames:
            param_dict[comp_name] = getattr(
                self._model, comp_name).parameterNames

        idx_param = 0
        for comp_name, param_names in param_dict.items():
            component = getattr(self._model, comp_name)
            for param_name in param_names:
                setattr(component, param_name, params[idx_param])
                idx_param += 1

    @classmethod
    def set_energy(cls, energy_kev_start, energy_kev_end, num_bands):
        xspec.AllModels.setEnergies(
             f"{energy_kev_start} {energy_kev_end} {num_bands}")

    @classmethod
    def set_chatter(cls, level):
        xspec.Xset.chatter = level

    @property
    def model(self):
        return self._model
