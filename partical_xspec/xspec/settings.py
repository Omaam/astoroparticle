"""Xspec settings.
"""
import os

import xspec


def set_energy(energy_kev_start, energy_kev_end, num_bands):
    xspec.AllModels.setEnergies(
        f"{energy_kev_start} {energy_kev_end} {num_bands}")


def set_response(response_file=None):
    # TODO (Tomoki): This does not work. Maybe no implement.
    xspec.AllModels.setEnergies(response_file)


def set_response_satellite_default(satellite_name="nicer"):

    satellite_name = satellite_name.lower()

    dirname = os.path.dirname(__file__)
    if satellite_name == "nicer":
        response_file = os.path.join(
            dirname, "nixtiref20170601v003.rmf")
    else:
        raise NotImplementedError()

    set_response(response_file)
