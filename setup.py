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
<<<<<<< HEAD
        "astropy",
=======
        "attrs",
>>>>>>> 0bfca58a6afd70a756b50a0ca6e2064c3301e2b8
        "numpy",
        "scipy",
        "lmfit",
        "statsmodels",
        "numba",
        "sklearn",
        "matplotlib"
    ],
   
)
