"""
'errors.py' contains custom error classes used
during image processing.
"""

class ImageError(Exception):
    """
    An exception class to handle errors during
    image processing.
    """

    def __init__(self, *args):
        """
        Handle a custom input message, if necessary.
        """
        if args:
            try:
                self.message = args[0]
            except TypeError:
                raise TypeError('Argument must be of type str().')
        else:
            self.message = None

    def __str__(self):
        """
        Define how the exception is displayed.
        """
        if self.message:
            return 'ImageError, {0}'.format(self.message)
        else:
            return 'ImageError, failed to process image.'


class ConfigError(Exception):
    """
    An exception class to handle errors during
    config parsing.
    """

    def __init__(self, *args):
        """
        Handle a custom input message, if necessary.
        """
        if args:
            try:
                self.message = args[0]
            except TypeError:
                raise TypeError('Argument must be of type str().')
        else:
            self.message = None

    def __str__(self):
        """
        Define how the exception is displayed.
        """
        if self.message:
            return f'ConfigError, {self.message}.'
        else:
            return 'ConfigError, failed to parse config file.'