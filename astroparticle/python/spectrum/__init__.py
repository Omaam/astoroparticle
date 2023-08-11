"""Observation class."""
from astroparticle.python import spectrum
from astroparticle.python.spectrum.binning import Rebin
from astroparticle.python.spectrum.response import CustumResponseModel
from astroparticle.python.spectrum.response_satellite import ResponseNicerXti
from astroparticle.python.spectrum.response_satellite import ResponseXrismResolve
from astroparticle.python.spectrum.components.powerlaw import PowerLaw
from astroparticle.python.spectrum.components.diskbb import DiskBB
from astroparticle.python.spectrum.components.diskpbb import DiskPBB
from astroparticle.python.spectrum.components.gauss import Gauss
from astroparticle.python.spectrum.components.physical_component import PhysicalComponent
