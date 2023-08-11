"""Energy spectrum component model module.
"""
import tensorflow as tf

from astroparticle.python.spectrum.components.physical_component \
    import PhysicalComponent


JIT_COMPILE = True


@tf.function(autograph=False, jit_compile=JIT_COMPILE)
def dkbflx(tin, photon_index, energy, dtype=tf.float32):
    """
    calculate the flux of the disk blackbody model at a given energy.

    args:
        tin (float): inner temperature of the disk (kev).
        p (float): exponent for the temperature radial
            dependence (t(r) ~ r^-p).
        energy (float): energy (kev).

    returns:
        float: photon flux in photons/s/cm^2/kev.
    """
    gauss = tf.constant(
        [0.236926885, 0.478628670, 0.568888888, 0.478628670,
         0.236926885, -0.906179846, -0.538469310, 0.0,
         0.538469310, 0.906179846], dtype=dtype)

    # nnn determines the accuracy of the numerical integration
    # and the time of the calculation.
    # When the minimum enegy is too low, it sometimes raise
    # memory allocation error. Consider employ adapting `nnn`
    # depending on energy and tin.
    energy_min = tf.reduce_min(energy)
    tin_min = tf.reduce_min(tin)
    nnn = 10000
    nnn = tf.cond(energy_min > 0.001 * tin_min, lambda: 1000, lambda: nnn)
    nnn = tf.cond(energy_min > 0.01 * tin_min, lambda: 500, lambda: nnn)
    nnn = tf.cond(energy_min > 0.1 * tin_min, lambda: 100, lambda: nnn)
    nnn = tf.cond(energy_min > 0.2 * tin_min, lambda: 10, lambda: nnn)
    nnn = tf.cond(energy_min > tin_min, lambda: 5, lambda: nnn)

    xn = 1.0 / tf.cast(nnn, dtype) / 2.0

    gauss_left = gauss[0:5]
    gauss_right = gauss[5:10]

    xh = xn * (2. * tf.range(1, nnn+1, 1, dtype=dtype) - 1)
    x = xn * gauss_right[tf.newaxis, :] + xh[..., tf.newaxis]

    energy_integ = energy[tf.newaxis, ..., tf.newaxis, tf.newaxis]
    tin_integ = tin[..., tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    photon_index_integ = photon_index[..., tf.newaxis, tf.newaxis,
                                      tf.newaxis, tf.newaxis]
    dk = tf.where(
        energy_integ / tin_integ / x >= 170,
        x ** (-2.0 / photon_index_integ - 1.0) * tf.exp(
            -energy_integ / tin_integ / x),
        x ** (-2.0 / photon_index_integ - 1.0) / (tf.exp(
            energy_integ / tin_integ / x) - 1.0)
    )
    photon = tf.reduce_sum(gauss_left * dk * xn, axis=[-2, -1])

    energy_batch = energy[tf.newaxis, :, :]
    photon_index_batch = photon_index[..., tf.newaxis, tf.newaxis]
    photons = 2.78e-3 * energy_batch * energy_batch * photon * (
        0.75 / photon_index_batch)

    return photons


@tf.function(autograph=False, jit_compile=JIT_COMPILE)
def diskpbb(energy_edges, num_energies, param, dtype=tf.float32):
    """
    compute the multicolor disk blackbody model spectrum.

    args:
        energy_edges (array-like): energy bin edges (kev).
        num_energies (int): number of energy bins.
        param (array-like): model parameters [tin (kev), p].

    returns:
        tuple: tuple containing arrays of photon counts in each energy
            bin (photons) and errors (photon_errors).
    """
    gauss = tf.constant(
        [0.236926885, 0.478628670, 0.568888888, 0.478628670,
         0.236926885, -0.906179846, -0.538469310, 0.0,
         0.538469310, 0.906179846], dtype=dtype)

    photon_errors = tf.zeros(num_energies, dtype=dtype)

    tin, p = tf.unstack(param, axis=-1)

    energy_edges_shifted = energy_edges[1:]
    energy_edges_prev = energy_edges[:-1]
    xn = (energy_edges_shifted - energy_edges_prev) / 2.0

    # xh.shape = (energy_edges, 2)
    xh = xn + energy_edges_prev
    gauss_left = gauss[0:5]
    gauss_right = gauss[5:10]
    energy = xn[:, tf.newaxis] * gauss_right[..., tf.newaxis, :] + \
        xh[:, tf.newaxis]

    photon = tf.reduce_sum(
        gauss_left[..., tf.newaxis, :] * dkbflx(tin, p, energy),
        axis=-1)
    photons = photon * xn

    return photons, photon_errors


class DiskPBB(PhysicalComponent):
    def __init__(self,
                 energy_edges,
                 tin=1.0,
                 photon_index=0.75,
                 normalization=1.0,
                 dtype=tf.float32,
                 name="diskpbb"):

        with tf.name_scope(name) as name:
            self.tin = tf.convert_to_tensor(
                tin, dtype=dtype)
            self.photon_index = tf.convert_to_tensor(
                photon_index, dtype=dtype)
            self.normalization = tf.convert_to_tensor(
                normalization, dtype=dtype)
            self.dtype = dtype
            super(DiskPBB, self).__init__(
                energy_edges_input=energy_edges,
                energy_edges_output=energy_edges)

    def _forward(self, flux):
        """Forward to calculate flux.
        """
        # TODO: Many uses of `tf.newaxis` make a mess.
        # Find another tider way.
        energy_edges = self.energy_edges_input
        tin = self.tin
        photon_index = self.photon_index
        normalization = self.normalization

        num_energies = energy_edges.shape[0] - 1
        param = tf.stack([tin, photon_index], axis=1)

        flux_diskpbb = normalization[:, tf.newaxis] * diskpbb(
            energy_edges, num_energies, param, dtype=self.dtype)[0]

        flux = flux + flux_diskpbb
        return flux

    def _set_parameter(self, x):
        x = tf.unstack(x, axis=-1)
        self.tin = x[0]
        self.photon_index = x[1]
        self.normalization = x[2]
