"""DiskPBB test.
"""
import tensorflow as tf

from astroparticle.python.test.xspec import XspecModel
from astroparticle.python.spectrum.components.phabs import Phabs
from astroparticle.python.spectrum.components.powerlaw import PowerLaw
from astroparticle.python.spectrum.components.testtool import XspecTester
from astroparticle.python.spectrum.components.sequence\
    import SequenceMultiplicative


class _PhabsTest(tf.test.TestCase, XspecTester):

    def testFlux(self):

        energy_edges = tf.linspace(0.1, 20.0, 3452)

        self.energy_edges = energy_edges
        self.component_xs = XspecModel("phabs*powerlaw")

        powerlaw = PowerLaw(energy_edges)
        phabs = Phabs(energy_edges)
        self.component_ap = SequenceMultiplicative([powerlaw, phabs])

        params = [[1.0, 1.0, 1.0]]

        # Comment out this line when you want to see spectra.
        # (default: desappear)
        # self.plot_two_fluxes(params)

        self.compare_flux(params)


class PhabsTestDynamicShape32(_PhabsTest):
    dtype = tf.float32


del _PhabsTest


if __name__ == "__main__":
    tf.test.main()
