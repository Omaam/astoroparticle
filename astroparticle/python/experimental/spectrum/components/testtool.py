"""Default comparison tester.
"""
import matplotlib.pyplot as plt
import tensorflow as tf

from astroparticle.python.test.xspec import XspecModel


class XspecTester:

    def compare_flux(self, params, assert_err=None):
        energy_ranges = [self.energy_edges[0], self.energy_edges[-1]]
        num_energies = self.energy_edges.shape[0] - 1

        XspecModel.set_energy(*energy_ranges, num_energies)
        self.component_xs.set_parameters(params[0])
        flux_xs = self.component_xs.compute_flux()

        self.component_ap.set_parameter(params)
        flux_ap = self.component_ap(tf.zeros(num_energies))[0]

        self.flux_xs = flux_xs
        self.flux_ap = flux_ap

        if assert_err is not None:
            self.assertArrayNear(flux_xs, flux_ap, assert_err)
        else:
            import warnings
            warnings.warn(
                "assertion not executed since `assert_err` is not given.")

    def _plot_two_fluxes(self):

        try:
            energy_edges = self.energy_edges
            flux_xs = self.flux_xs
            flux_ap = self.flux_ap
        except AttributeError:
            raise ValueError(
                "you must execute 'compare_flux()' before plotting.")

        fig, ax = plt.subplots()
        energy_centers = (energy_edges[1:] + energy_edges[:-1]) / 2
        ax.plot(energy_centers, flux_xs, color="k")
        ax.plot(energy_centers, flux_ap, color="r")
        ax.set_xlabel("energy (keV)")
        ax.set_ylabel("photon/s/cm2")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.show()
        plt.close()
