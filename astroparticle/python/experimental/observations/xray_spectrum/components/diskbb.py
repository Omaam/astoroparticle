"""Energy spectrum component model module.
"""
import tensorflow as tf

from astroparticle.python.experimental.observations.\
    xray_spectrum.components.diskpbb import DiskPBB


class DiskBB(DiskPBB):
    """Disk black body model

    This model inherites `apo.DiskPBB` model with `photon_index=0.75`.
    """
    def __init__(self,
                 energy_intervals_input,
                 tin=1.0,
                 normalization=1.0,
                 dtype=tf.float32,
                 name="diskbb"):

        with tf.name_scope(name) as name:
            batch_shape = tin.shape
            photon_index = tf.repeat(tf.constant(0.75, dtype=dtype),
                                     batch_shape)
            super(DiskBB, self).__init__(
                 energy_intervals_input,
                 tin=tin,
                 photon_index=photon_index,
                 normalization=normalization,
                 dtype=dtype,
                 name=name)

    def _set_parameter(self, x):
        x = tf.unstack(x, axis=-1)
        self.tin = x[0]
        self.normalization = x[1]
