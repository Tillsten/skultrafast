# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:46:54 2013

@author: tillsten
"""

import numpy as np
import time

from traits.etsconfig.api import ETSConfig

ETSConfig.toolkit = 'qt4'

# ETS imports (non-chaco)
from enable.component_editor import ComponentEditor
from enable.api import BaseTool
from traits.api import HasTraits, Instance, Int, List, Str, Enum,\
    on_trait_change, Any, DelegatesTo, Array, Button, Float, Bool, Function


from traitsui.api import Item, View, HSplit, HGroup,\
    EnumEditor, Group, UItem, VGroup, Include, InstanceEditor

# Chaco imports
from chaco.api import ArrayPlotData, Plot, ColorBar, HPlotContainer
import chaco.api as dc

from chaco.example_support import COLOR_PALETTE
from chaco.tools.api import PanTool, ZoomTool, RangeSelection,\
    RangeSelectionOverlay, LegendTool

import sklearn.linear_model as lm
from skultrafast.dv import tup
import skultrafast.lifetimemap as ltm

class LmParams(HasTraits):
    data = Instance(tup)
    n_exps = Int(50, low=1, 
                 tooltip='Number of exponentials used as base')
    max_iter = Int(10000)
    
    params = Group()
    view = View(['max_iter', 'n_exps', Include('params')])
    
    
    def fit(self):      
        base = ltm._make_base(self.data)
        self.fitobj.fit()

    
    
class LassoParams(LmParams):
    alpha = Float(0.001)    
    fitobj = Instance(lm.Lasso)
    params = Group(Item('alpha'))
        
    def _fitobj_default(self):
        return llm.Lasso(alpha = self.alpha, max_iter=self.max_iter)

    
class LassoCVParams(LmParams):
    n_alphas = Int(5)
    fitobj = Instance(lm.LassoCV)
    params = Group(['n_alphas'])
    
    def _fitobj_default(self):
        return lm.LassoCV(n_alphas = self.n_alphas, mat_iter=self.max_iter)
    

method_dict = {'Lasso': LassoParams, 
               'LassoCV': LassoCVParams,
               }   

class LifeTimeMapTool(HasTraits):
    method = Enum('Lasso', 'LassoCV')
                #  'ElasticNet', 'ElasticNetCV', 'MultiTaskElasticNet')
                  
    but_fit = Button('Fit')
    lm = Instance(LmParams)
    data = Instance(tup)    
    
    img_plot = Instance(dc.Plot)
    
    def _lm_default(self):
        return LassoParams(data=self.tup)
    
    def _img_plot_default(self):
        self._plot_data = ArrayPlotData(signal=self.tup.data)
        print self.tup.data
        plot = dc.Plot(self._plot_data)
        plot.img_plot('signal')
        return plot
        
    def _method_changed(self):
        self.lm = method_dict[self.method](data=self.tup)
        
    def _but_fit_fired(self):
        self.lm.fit()
        

    para_group = VGroup(Item('lm', style='custom', show_label=False),
                  Item('method', style='custom'),
                  Item('but_fit', show_label=False),
                  )
                  
    plot_item = Item('img_plot', editor=ComponentEditor(), show_label=False)
    view = View(HGroup(para_group, plot_item ))
#l = LassoParams(tup=k).configure_traits()

if __name__ == '__main__':
    import pickle
    k = pickle.load(file('test_tup'))    
    l = LifeTimeMapTool(tup=k).configure_traits()
    
