# -*- coding: utf-8 -*-
from setuptools import setup
import versioneer

setup(
    cmdclass=versioneer.get_cmdclass(),
    name='skultrafast',
    version=versioneer.get_version(),
    author='Till Stensitzki',
    author_email='tillsten@zedat.fu-berlin.de',
    url='http://github.com/tillsten/skultrafast-git',
    packages=['skultrafast',],
    license='LICENSE.txt',
    description='Python package for analyzing time-resolved spectra.',
    long_description=open('README.rst').read(),
    install_requires=[
        "astropy",
        "attrs",
        "numpy",
        "scipy",
        "lmfit",
        "statsmodels",
        "numba",
        "sklearn",
        "matplotlib"
    ],
    keywords='science physics chemistry pump-probe spectroscopy time-resolved',
    python_requires='>=3.5',
)
