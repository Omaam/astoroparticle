"""Energy spectrum model base module.
"""
import tensorflow as tf


class XraySpectrum(tf.Module):
    def __init__(self,
                 energy_intervals_input,
                 energy_intervals_output):
        self._energy_intervals_input = energy_intervals_input
        self._energy_intervals_output = energy_intervals_output

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

    @property
    def energy_intervals_output(self):
        return self._energy_intervals_output

    @property
    def energy_size_input(self):
        return self._energy_intervals_input.shape[0]

    @property
    def energy_size_output(self):
        return self._energy_intervals_output.shape[0]
