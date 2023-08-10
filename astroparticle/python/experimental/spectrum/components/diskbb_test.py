"""DiskBB test.
"""
import tensorflow as tf

from astroparticle.python.test.xspec import XspecModel
from astroparticle.python.experimental.spectrum.components.diskbb \
    import DiskBB
from astroparticle.python.experimental.spectrum.components.testtool \
    import XspecTester


class _DiskBBTest(tf.test.TestCase, XspecTester):

    def testFlux(self):

        energy_edges = tf.linspace(0.1, 20.0, 3452)

        self.energy_edges = energy_edges
        self.component_xs = XspecModel("diskbb")
        self.component_ap = DiskBB(energy_edges)

        params = [[1.0, 1e6]]
        self.compare_flux(params, assert_err=1.)

        # Cancel comentout when you see spectra.
        # self._plot_two_fluxes()


class DiskBBTestDynamicShape32(_DiskBBTest):
    dtype = tf.float32


del _DiskBBTest


if __name__ == "__main__":
    tf.test.main()
