"""Energy spectrum model base module.
"""
import tensorflow as tf

from core import XraySpectrum
from components.powerlaw import PowerLaw
from response_satellite import ResponseNicerXti
from binning import Rebin


class MyModel(XraySpectrum):
    def __init__(self):
        response_nicer_xti = ResponseNicerXti()
        energy_edges = response_nicer_xti.energy_edges

        spectrum_params = tf.random.normal(
            (100, 2), mean=2.0, stddev=0.01)
        self.powerlaws = PowerLaw(energy_edges,
                                  photon_index=spectrum_params[:, 0],
                                  normalization=spectrum_params[:, 1])

        self.response = response_nicer_xti
        self.rebin = Rebin(energy_splits_old=tf.linspace(0.1, 20., 1494),
                           energy_splits_new=tf.linspace(0.1, 20., 11))

    def __call__(self, flux):
        flux = self.powerlaws(flux)
        flux = self.response(flux)
        flux = self.rebin(flux)
        return flux


def main():

    mymodel = MyModel()
    flux = mymodel(tf.zeros(3451, dtype=tf.float32))
    print(flux.shape)


if __name__ == "__main__":
    main()
