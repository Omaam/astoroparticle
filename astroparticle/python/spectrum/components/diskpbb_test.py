"""DiskPBB test.
"""
import tensorflow as tf

from astroparticle.python.spectrum.components.diskpbb \
    import DiskPBB
from astroparticle.python.spectrum.components.testtool \
    import XspecTester
from astroparticle.python.test.xspec import XspecModel


class _DiskPBBTest(tf.test.TestCase, XspecTester):

    def testFlux(self):
        energy_edges = tf.linspace(0.1, 20.0, 3452)

        self.energy_edges = energy_edges
        self.component_xs = XspecModel("diskpbb")
        self.component_ap = DiskPBB(energy_edges)

        params = [[1.0, 0.75, 1e6]]
        self.compare_flux(params)

        # Cancel comentout when you see spectra.
        # self.plot_two_fluxes()


class DiskPBBTestDynamicShape32(_DiskPBBTest):
    dtype = tf.float32


del _DiskPBBTest


if __name__ == "__main__":
    tf.test.main()
