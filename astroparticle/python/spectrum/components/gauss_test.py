"""DiskPBB test.
"""
import tensorflow as tf

from astroparticle.python.test.xspec import XspecModel
from astroparticle.python.spectrum.components.gauss \
    import Gauss
from astroparticle.python.spectrum.components.testtool \
    import XspecTester


class _GaussTest(tf.test.TestCase, XspecTester):

    def testFlux(self):

        energy_edges = tf.linspace(0.1, 20.0, 3452)

        self.energy_edges = energy_edges
        self.component_xs = XspecModel("gauss")
        self.component_ap = Gauss(energy_edges)

        params = [[6.4, 0.1, 10.]]
        self.compare_flux(params, assert_err=0.01)

        # Cancel comentout when you see spectra.
        # self._plot_two_fluxes()


class GaussTestDynamicShape32(_GaussTest):
    dtype = tf.float32


del _GaussTest


if __name__ == "__main__":
    tf.test.main()
