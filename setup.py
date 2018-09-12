# -*- coding: utf-8 -*-


from setuptools import setup

setup(
    name='skultrafast',
    version='1.0',
    author='Till Stensitzki',
    author_email='tillsten@zedat.fu-berlin.de',
    packages=['skultrafast',],
    license='LICENSE.txt',
    description='Python package for analyzing time-resolved spectra',
    #long_description=open('README.rst').read(),
    install_requires=[
        "astropy",
        "numpy",
        "scipy",
        "lmfit",
        "statsmodels",
        "numba",
        "sklearn",
        "matplotlib"
    ],
   
)