"""
Distribution functions for target generation in the Child Mind Institute - Detect Sleep States project
"""

from math import exp, gamma, pi, sqrt

import numpy as np


def gauss(n=720, sigma=108):
    """
    Gaussian distribution function

    Args:
        n: Size of the distribution (default: 720)
        sigma: Standard deviation (default: 108)

    Returns:
        List of Gaussian distribution values
    """
    r = range(-int(n / 2), int(n / 2) + 1)
    return [1 / (sigma * sqrt(2 * pi)) * exp(-(float(x) ** 2) / (2 * sigma**2)) for x in r]


def gauss_standard(n=720, sigma=108):
    """
    Standardized Gaussian distribution function

    Args:
        n: Size of the distribution (default: 720)
        sigma: Standard deviation (default: 108)

    Returns:
        Numpy array of standardized Gaussian distribution values
    """
    r = range(-int(n / 2), int(n / 2) + 1)
    r = [e / sigma for e in r]
    return np.array([1 / (sigma * sqrt(2 * pi)) * exp(-(float(x) ** 2) / 2) for x in r])


def lognorm(n=720, sigma=108):
    """
    Log-normal distribution function

    Args:
        n: Size of the distribution (default: 720)
        sigma: Standard deviation (default: 108)

    Returns:
        Numpy array of log-normal distribution values
    """
    r = range(-int(n / 2), int(n / 2) + 1)
    r = [e / sigma for e in r]
    return np.exp(np.array([1 / (sigma * sqrt(2 * pi)) * exp(-(float(x) ** 2) / 2) for x in r]))


def lognormal_standard(n=720, sigma=82.8):
    """
    Standardized log-normal distribution function

    Args:
        n: Size of the distribution (default: 720)
        sigma: Standard deviation (default: 82.8)

    Returns:
        Numpy array of standardized log-normal distribution values
    """
    # Define the range of x, ensuring all values are positive and not zero
    r = range(1, n + 2, 1)  # Avoid zero because ln(0) is not defined
    r = np.array([e / sigma for e in r])
    # Calculate the PDF of the standard log-normal distribution
    pdf = (1 / (r * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(r)) ** 2))
    return np.concatenate([pdf[::-1][331:], np.zeros(242 + 89)])


def student_t(n=720, v=2.000171482465918, sigma=108):
    """
    Student's t-distribution function

    Args:
        n: Size of the distribution (default: 720)
        v: Degrees of freedom (default: 2.000171482465918)
        sigma: Standard deviation (default: 108)

    Returns:
        Numpy array of Student's t-distribution values
    """
    # Student's t-distribution function
    r = range(-int(n / 2), int(n / 2) + 1)
    r = [e / sigma for e in r]
    # Calculate the multiplier term outside the sum which is constant for given v
    multiplier = 1 / sigma * gamma((v + 1) / 2) / (sqrt(v * pi) * gamma(v / 2))
    return np.array([multiplier * (1 + (float(x) ** 2 / v)) ** (-(v + 1) / 2) for x in r])
