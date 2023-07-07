"""Xspec settings.
"""
import xspec


def set_energy(energy_kev_start, energy_kev_end, num_bands):
    xspec.AllModels.setEnergies(
        f"{energy_kev_start} {energy_kev_end} {num_bands}")
