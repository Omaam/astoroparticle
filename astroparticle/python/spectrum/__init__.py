"""Observation class."""
from astroparticle.python.experimental import spectrum
from astroparticle.python.experimental.spectrum.binning import Rebin
from astroparticle.python.experimental.spectrum.response import CustumResponseModel
from astroparticle.python.experimental.spectrum.response_satellite import ResponseNicerXti
from astroparticle.python.experimental.spectrum.response_satellite import ResponseXrismResolve
from astroparticle.python.experimental.spectrum.components.powerlaw import PowerLaw
from astroparticle.python.experimental.spectrum.components.diskbb import DiskBB
from astroparticle.python.experimental.spectrum.components.diskpbb import DiskPBB
from astroparticle.python.experimental.spectrum.components.gauss import Gauss
from astroparticle.python.experimental.spectrum.components.physical_component import PhysicalComponent
