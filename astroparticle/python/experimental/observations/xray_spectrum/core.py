"""Energy spectrum model base module.
"""
import tensorflow as tf


class XraySpectrum(tf.Module):
    def __init__(self, dtype=tf.float32):
        self.dtype = dtype

    def __call__(self, value):
        return self.forward(value)

    def _forward(self, x):
        """Subclass implementation for `forward` public function."""
        raise NotImplementedError("forward not implemented.")

    def forward(self, flux):
        return self._forward(flux)

    @property
    def energy_intervals_input(self):
        return self._energy_intervals_input

    def _energy_intervals_input(self):
        raise NotImplementedError(
            "energy_intervals_input not implemented.")

    @property
    def energy_intervals_output(self):
        return self._energy_intervals_output

    @property
    def _energy_intervals_output(self):
        raise NotImplementedError(
            "energy_intervals_output not implemented.")
