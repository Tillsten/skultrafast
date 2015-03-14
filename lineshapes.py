# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:17:07 2013

@author: Tillsten
"""
import numpy  as np

def lorentz(x, A, w, xc):
    return A/(1 + ((x-xc)/w)**2)

def gaussian(x, A, w, xc):
    return A*np.exp(((x-xc)/w)**2)
    