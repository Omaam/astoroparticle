"""Transition class.
"""


class Transition:
    def __init__():
        pass

    @property
    def default_latent_indicies(self):
        return self._default_latent_indicies()

    def _default_latent_indicies(self):
        raise NotImplementedError(
            "_parameter_properties` is not implemented: {}.".format(
                self.__name__))

    def get_function(self, **kwargs):
        return self._get_function(**kwargs)

    def _get_function(self, **kwargs):
        raise NotImplementedError(
            "_parameter_properties` is not implemented: {}.".format(
                self.__name__))
