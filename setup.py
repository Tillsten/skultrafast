# -*- coding: utf-8 -*-


from setuptools import setup

setup(
    name='skultrafast',
    version='0.8.0',
    author='Till Stensitzki',
    author_email='tillsten@zedat.fu-berlin.de',
    packages=['skultrafast',],
    license='LICENSE.txt',
    description='Python package for analyzing time-resolved spectra',
    #long_description=open('README.rst').read(),
    install_requires=[
        "attrs",
        "numpy",
        "scipy",
        "lmfit",
        "statsmodels",
        "numba",
        "sklearn",
        "matplotlib"
    ],
   
)
