"""Transition class.
"""


class Transition:
    def __init__():
        pass

    @property
    def transition_function(self, **kwargs):
        return self._transition_function(**kwargs)

    def _transition_function(self, **kwargs):
        raise NotImplementedError(
            "_parameter_properties` is not implemented: {}.".format(
                self.__name__))
