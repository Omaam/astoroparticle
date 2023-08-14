"""Energy spectrum component model module.
"""
import os

import numpy as np
import tensorflow as tf

from astroparticle.python.spectrum.components.physical_component \
    import PhysicalComponent


def xszabs_bard(energy_edges, param):
    """
    calculate the absorption due to the xszabs model.

        Args:
            energy_edges: An array of energies in keV.
            ne: The number of elements in the energy_edges array.
            param: An array of model parameters, in the order [NH, z].

        Returns:
            photar, photer: The updated arrays of the transmitted fluxes
                and their errors.

        Raises:
            ValueError: If the number of model parameters is not 2.

    """

    edge = np.array([0.030, 0.100, 0.284, 0.400, 0.532, 0.707, 0.867, 1.303,
                     1.840, 2.471, 3.210, 4.038, 7.111, 8.331, 10.00, 1e10])
    c0 = np.array([0.336, 0.173, 0.346, 0.781, 0.714, 0.955, 3.089, 1.206,
                   1.413, 2.027, 3.427, 3.522, 4.339, 6.290, 7.012, 9.532])
    c1 = np.array([0.000, 6.081, 2.679, 0.188, 0.668, 1.458, -3.806, 1.693,
                   1.468, 1.047, 0.187, 0.187, -0.024, 0.309, 0.252, 0.000])
    c2 = np.array([0.000, -21.5, -4.761, 0.043, -0.514, -0.611, 2.94, -0.477,
                   -0.315, -0.17, 0.000, 0.000, 0.0075, 0.000, 0.000, 0.000])

    anh = param[0]
    zfac = 1.0 + param[1]

    # Find edge energy just greater than initial array energy
    ir = 0
    while edge[ir] <= energy_edges[0] * zfac:
        ir += 1

    cedge = edge[ir]
    cc2 = c2[ir]
    cc1 = c1[ir]
    cc0 = c0[ir]
    oldf = 0.0
    eold = 0.0
    oldenergy_edges = energy_edges[0] * zfac

    num_energies = energy_edges.shape[-1] - 1
    photar = np.zeros(num_energies)
    photer = np.zeros(num_energies)

    for ie in range(num_energies):
        cenergy_edges = energy_edges[ie] * zfac
        e = min(cenergy_edges, cedge)
        qlaste = False
        facum = 0.0
        while not qlaste:
            qedge = e == cedge
            qlaste = (not qedge) and (e == cenergy_edges)

            # Evaluate the absorption at e
            if e < 0.0136:
                cf = 1.0
            else:
                einv = 1.0 / e
                sigma = einv * (cc2 + einv * (cc1 + einv * cc0))
                cf = np.exp(-anh * sigma)

            if qedge or qlaste:
                facum += (e - eold) * (oldf + cf)
                eold = e
                if qedge:
                    ir += 1
                    cedge = edge[ir]
                    cc2 = c2[ir]
                    cc1 = c1[ir]
                    cc0 = c0[ir]
                    qedge = False
                else:
                    # This can be reached only if qlaste is true
                    oldf = cf

            else:
                # Since it is not the end of an edge OR the end of the
                # energy_edges range, then this must be the initial part of the
                # energy range for a new edge
                eold = e
                oldf = cf
                e = min(cedge, cenergy_edges)

        if ie != 0:
            if cenergy_edges != oldenergy_edges:
                photar[ie] = facum * 0.5 / (cenergy_edges - oldenergy_edges)
            else:
                photar[ie] = 0.0

        oldenergy_edges = cenergy_edges

    return photar, photer


def xszabs_chatgpt(ear, param):
    ne = len(ear) - 1
    photar = np.zeros(ne)
    photer = np.zeros(ne)

    NRANGE = 16
    edge = np.array([0.03, 0.1, 0.284, 0.4, 0.532, 0.707, 0.867, 1.303,
                     1.840, 2.471, 3.210, 4.038, 7.111, 8.331, 10., 1.e10])
    c0 = np.array([0.336, 0.173, 0.346, 0.781, 0.714, 0.955, 3.089,
                   1.206, 1.413, 2.027, 3.427, 3.522, 4.339, 6.29, 7.012,
                   9.532])
    c1 = np.array([0., 6.081, 2.679, 0.188, 0.668, 1.458, -3.806, 1.693,
                   1.468, 1.047, 0.187, 0.187, -0.024, 0.309, 0.252, 0.])
    c2 = np.array([0., -21.5, -4.761, 0.043, -0.514, -0.611, 2.94, -0.477,
                   -0.315, -0.17, 0., 0.0075, 0., 0., 0., 0.])

    anh = param[0]
    zfac = 1.0 + param[1]

    photer = np.zeros(ne)

    for ie in range(ne):
        cear = ear[ie] * zfac
        e = min(cear, edge[0])
        qlaste = False
        facum = 0.0

        for ir in range(1, NRANGE):
            cedge = edge[ir]
            cc2 = c2[ir]
            cc1 = c1[ir]
            cc0 = c0[ir]
            oldf = 0.0
            eold = 0.0

            while not qlaste:
                qedge = e == cedge
                qlaste = (not qedge) and (e == cear)

                if e < 0.0136:
                    cf = 1.0
                else:
                    einv = 1.0 / e
                    sigma = einv * (cc2 + (einv * (cc1 + (einv * cc0))))
                    cf = np.exp(-anh * sigma)

                if qedge or qlaste:
                    facum += (e - eold) * (oldf + cf)
                    eold = e
                    if qedge:
                        ir += 1
                        cedge = edge[ir]
                        cc2 = c2[ir]
                        cc1 = c1[ir]
                        cc0 = c0[ir]
                        qedge = False
                    else:
                        oldf = cf
                else:
                    eold = e
                    oldf = cf
                    e = min(cedge, cear)

        if ie != 0:
            if cear != ear[ie - 1] * zfac:
                photar[ie] = facum * 0.5 / (cear - ear[ie - 1] * zfac)

    return photar, photer


class Phabs(PhysicalComponent):
    def __init__(self,
                 energy_edges,
                 nh=1.0,
                 dtype=tf.float32,
                 name="Phabs"):
        with tf.name_scope(name) as name:
            energy_edges = tf.convert_to_tensor(energy_edges, dtype=dtype)
            nh = tf.convert_to_tensor(nh, dtype=dtype)

            super(Phabs, self).__init__(
                energy_edges_input=energy_edges,
                energy_edges_output=energy_edges)

            self.nh = nh
            self._parameter_size = 1
            self.dtype = dtype

    def _forward(self, flux):
        batch_shape = self.nh.shape[:-1]
        redshift = tf.broadcast_to(0.0, [*batch_shape, 1])
        param = np.stack([self.nh, redshift], axis=-1)
        new_flux = xszabs_bard(self.energy_edges_input, param[0])
        # new_flux = xszabs_chatgpt(self.energy_edges_input, param[0])
        flux = flux * new_flux[0]
        return flux

    def _set_parameter(self, x):
        x = tf.convert_to_tensor(x, dtype=self.dtype)
        x = tf.unstack(x, axis=-1)
        self.nh = x[0]


class PhabsNicerXti(PhysicalComponent):
    def __init__(self,
                 energy_edges,
                 nh=1.0,
                 dtype=tf.float32,
                 name="Phabs"):
        with tf.name_scope(name) as name:

            energy_edges = tf.convert_to_tensor(energy_edges, dtype=dtype)
            nh = tf.convert_to_tensor(nh, dtype=dtype)

            # The number of energies for NICER XTI is 3451, thus the one
            # of the energy edges are 3452.
            if energy_edges.shape[-1] != 3452:
                raise ValueError(
                    "`energy_edges.shape[-1]` must be 3452 if you want "
                    "to use this class"
                )

            super(PhabsNicerXti, self).__init__(
                energy_edges_input=energy_edges,
                energy_edges_output=energy_edges)

            phabs_data = tf.convert_to_tensor(
                np.loadtxt(os.path.join(os.path.dirname(__file__),
                                        "data/phabs_nicer.txt")),
                dtype=dtype
            )

            self.nh = nh
            self.phabs_data = phabs_data
            self._parameter_size = 1
            self.dtype = dtype

    def _forward(self, flux):
        phabs_values = self.nh[:, tf.newaxis] * self.phabs_data
        flux = flux * phabs_values
        return flux

    def _set_parameter(self, x):
        x = tf.convert_to_tensor(x, dtype=self.dtype)
        x = tf.unstack(x, axis=-1)
        self.nh = x[0]
