"""Gauss test.
"""
import tensorflow as tf

from astroparticle.python.experimental.observations.\
    xray_spectrum.components.gauss import Gauss


class GaussTest(tf.test.TestCase):
    DTYPE = tf.float32

    def test_shape(self):

        num_energies = 1000
        true_shape = (1, 1, num_energies)

        flux = tf.broadcast_to(tf.zeros(num_energies), true_shape)

        energy_intervals = self._get_energy_intervals(
            0.5, 10., num_energies+1)

        gauss = Gauss(energy_intervals)

        x = tf.constant([[[6.4, 0.1, 1.0]]], dtype=self.DTYPE)
        gauss.set_parameter(x)
        flux = gauss(flux)
        self.assertAlmostEqual(true_shape, flux.shape)

    def test_total_area(self):
        energy_intervals = self._get_energy_intervals(0.5, 10., 1001)
        flux = tf.zeros(energy_intervals.shape[-2])[tf.newaxis, :]

        gauss = Gauss(energy_intervals)
        x = tf.constant([[6.4, 0.1, 1.0]], dtype=self.DTYPE)
        gauss.set_parameter(x)
        flux = gauss(flux)
        sum_flux = tf.reduce_sum(flux, axis=-1)
        expected = tf.constant([1.], dtype=self.DTYPE)
        self.assertAlmostEqual(expected, sum_flux)

    def _get_energy_intervals(self, start, stop, num):
        energy_edges = tf.linspace(start, stop, num)
        energy_intervals = tf.stack(
            [energy_edges[:-1],
             energy_edges[1:]],
            axis=-1
        )
        return tf.cast(energy_intervals, self.DTYPE)


if __name__ == "__main__":
    tf.test.main()
