"""
This module contains the lmfit models used in the fitting of the data.
"""

import lmfit
import numpy as np


def amp_gauss(x, A, xc, fhwm):
    """
    Gaussian function with amplitude A, center xc and full width at half
    maximum fhwm.

    Parameters
    ----------
    x : array
        Independent variable
    A : float
        Amplitude of the Gaussian
    xc : float
        Center of the Gaussian
    fhwm : float
        Full width at half maximum

    Returns
    -------
    array
        The Gaussian function
    """
    sigma = fhwm / (2 * np.sqrt(2 * np.log(2)))
    return A * np.exp(-(x - xc)**2 / (2 * sigma**2))


def amp_lorentz(x, A, xc, fhwm):
    """
    Lorentzian function with amplitude A, center xc and full width at half
    maximum fhwm.

    Parameters
    ----------
    x : array
        Independent variable
    A : float
        Amplitude of the Lorentzian
    xc : float
        Center of the Lorentzian
    fhwm : float
        Full width at half maximum

    Returns
    -------
    array
        The Lorentzian function
    """
    gamma = fhwm / 2
    return A / (1 + ((x - xc) / gamma)**2)


class AmpGaussian(lmfit.Model):
    """
    A Gaussian model with amplitude as a parameter.
    """

    def __init__(self, *args, **kwargs):
        super(AmpGaussian, self).__init__(amp_gauss, *args, **kwargs)
        self.set_param_hint('A', value=1.0)
        self.set_param_hint('xc', value=0.0)
        self.set_param_hint('fhwm', value=1.0, min=0.0)

    def guess(self, data, x, **kws):
        """
        Guess the initial values of the parameters.

        Parameters
        ----------
        data : array
            The data to fit
        x : array
            The independent variable

        Returns
        -------
        lmfit.Parameters
            The initial values of the parameters
        """
        params = self.make_params()
        params['A'].value = data.max()
        params['xc'].value = x[np.argmax(data)]
        params['fhwm'].value = x.std()
        return params


class AmpLorentzian(lmfit.Model):
    """
    A Lorentzian model with amplitude as a parameter.
    """

    def __init__(self, *args, **kwargs):
        super(AmpLorentzian, self).__init__(amp_lorentz, *args, **kwargs)
        self.set_param_hint('A', value=1.0)
        self.set_param_hint('xc', value=0.0)
        self.set_param_hint('fhwm', value=1.0, min=0.0)

    def guess(self, data, x, **kws):
        """
        Guess the initial values of the parameters.

        Parameters
        ----------
        data : array
            The data to fit
        x : array
            The independent variable

        Returns
        -------
        lmfit.Parameters
            The initial values of the parameters
        """
        params = self.make_params()
        params['A'].value = data.max()
        params['xc'].value = x[np.argmax(data)]
        params['fhwm'].value = x.std()
        return params
