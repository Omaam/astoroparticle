"""DiskPBB test.
"""
import tensorflow as tf

from astroparticle.python.test.xspec import XspecModel
from astroparticle.python.experimental.spectrum.components.powerlaw \
    import PowerLaw
from astroparticle.python.experimental.spectrum.components.testtool \
    import XspecTester


class _PowerLawTest(tf.test.TestCase, XspecTester):

    def testFlux(self):

        energy_edges = tf.linspace(0.1, 20.0, 3452)

        self.energy_edges = energy_edges
        self.component_xs = XspecModel("powerlaw")
        self.component_ap = PowerLaw(energy_edges)

        params = [[1.0, 10.]]
        self.compare_flux(params, assert_err=1.)

        # Cancel comentout when you see spectra.
        # self._plot_two_fluxes()


class PowerLawTestDynamicShape32(_PowerLawTest):
    dtype = tf.float32


del _PowerLawTest


if __name__ == "__main__":
    tf.test.main()
