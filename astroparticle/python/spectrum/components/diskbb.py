"""Energy spectrum component model module.
"""
import tensorflow as tf

from astroparticle.python.experimental.spectrum.components.diskpbb \
    import DiskPBB


class DiskBB(DiskPBB):
    """Disk black body model

    This model inherites `apo.DiskPBB` model with `photon_index=0.75`.
    """
    def __init__(self,
                 energy_edges,
                 tin=1.0,
                 normalization=1.0,
                 dtype=tf.float32,
                 name="diskbb"):

        tin = tf.convert_to_tensor(tin, dtype=dtype)
        normalization = tf.convert_to_tensor(normalization, dtype=dtype)

        with tf.name_scope(name) as name:
            batch_shape = tin.shape
            photon_index = tf.broadcast_to(0.75, batch_shape)
            super(DiskBB, self).__init__(
                 energy_edges,
                 tin=tin,
                 photon_index=photon_index,
                 normalization=normalization,
                 dtype=dtype,
                 name=name)

    def _set_parameter(self, x):
        x = tf.convert_to_tensor(x, dtype=self.dtype)
        batch_shape = x.shape[:-1]
        x = tf.unstack(x, axis=-1)
        x = tf.stack(
            [x[0],
             tf.broadcast_to(0.75, batch_shape),
             x[1]],
            axis=-1)

        # TODO: Is this writing style OK?
        super(DiskBB, self)._set_parameter(x)
