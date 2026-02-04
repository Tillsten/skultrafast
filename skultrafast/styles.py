# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 21:33:24 2015

@author: Tillsten
"""

import matplotlib.pyplot as plt
import numpy as np

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

tableau20 = [(r/255., g/255., b/255.) for r,g,b, in tableau20]

#plt.rcParams['savefig.dpi'] = 110
#plt.rcParams['font.family'] = 'Vera Sans'

out_ticks = {'xtick.direction': 'out',
             'xtick.major.width': 1.5,
             'xtick.minor.width': 1,
             'xtick.major.size': 6,
             'xtick.minor.size': 3,
             'xtick.minor.visible': True,
             'ytick.direction': 'out',
             'ytick.major.width': 1.5,
             'ytick.minor.width': 1,
             'ytick.major.size': 6,
             'ytick.minor.size': 3,
             'ytick.minor.visible': True,
             'axes.spines.top': False,
             'axes.spines.right': False,
             'text.hinting': True,
             'axes.titlesize': 'xx-large',
             'axes.titleweight': 'semibold',
             }

