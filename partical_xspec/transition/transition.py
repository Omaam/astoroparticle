"""Transition class.
"""


class Transition:
    def __init__():
        pass

    @property
    def default_using_latent_indicies(self):
        return self._default_using_latent_indicies()

    def _default_using_latent_indicies(self):
        raise NotImplementedError(
            "_parameter_properties` is not implemented: {}.".format(
                self.__name__))

    @property
    def transition_function(self, **kwargs):
        return self._transition_function(**kwargs)

    def _transition_function(self, **kwargs):
        raise NotImplementedError(
            "_parameter_properties` is not implemented: {}.".format(
                self.__name__))
