"""Default comparison tester.
"""
import matplotlib.pyplot as plt
import tensorflow as tf

from astroparticle.python.test.xspec import XspecModel


class XspecTester:

    def compare_flux(self, params, rtol=1e-02, atol=1e-4):
        flux_xs, flux_ap = self.compute_flux(params)
        self.assertAllClose(flux_xs, flux_ap, rtol=rtol, atol=atol)

    def compute_flux(self, params):
        energy_ranges = [self.energy_edges[0], self.energy_edges[-1]]
        num_energies = self.energy_edges.shape[0] - 1

        XspecModel.set_energy(*energy_ranges, num_energies)
        self.component_xs.set_parameters(params[0])
        flux_xs = self.component_xs.compute_flux()

        self.component_ap.set_parameter(params)
        flux_ap = self.component_ap(tf.zeros(num_energies))[0]

        return flux_xs, flux_ap

    def plot_two_fluxes(self, params):

        flux_xs, flux_ap = self.compute_flux(params)

        fig, ax = plt.subplots()
        energy_centers = (self.energy_edges[1:] + self.energy_edges[:-1]) / 2
        ax.plot(energy_centers, flux_xs, color="k")
        ax.plot(energy_centers, flux_ap, color="r")
        ax.set_xlabel("energy (keV)")
        ax.set_ylabel("photon/s/cm2")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.show()
        plt.close()
