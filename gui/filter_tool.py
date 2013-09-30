# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:01:13 2013

@author: tillsten
"""
import traits.api as tr
import traitsui.api as ui
import chaco.api as ch
import enable.api as en
import numpy as np

class Filter(tr.HasTraits):
    t = tr.Array
    d = tr.Array
    df = tr.Array
    
    replot = tr.Event
        
    def _df_default(self):        
        df = np.zeros_like(self.d)
        return df        

class SvdFilter(Filter):
    number_of_components = tr.Range(1, 20)
    
    
    @tr.on_trait_change('number_of_components')
    def apply_filter(self):        
        print "Hiier", self.d
        u, s, v = np.linalg.svd(self.d, full_matrices=False)
        s[self.number_of_components:] = 0
        s = np.diag(s)
        self.df = u.dot(np.dot(s, v))        
        self.replot = True
        return self.df

          
    traits_view = ui.View(['number_of_components'])
        

class FilterTool(tr.HasTraits):
    data = tr.Array
    t = tr.Array     
    
    _ds = tr.Instance(ch.ArrayPlotData)
    _plot = tr.Instance(ch.Plot)        

    def _filter_default(self):
        print "fil", self.data
        return SvdFilter(t=self.t, d=self.data)
        
#    df = tr.DelegatesTo('filter')
    filter = tr.Instance(Filter)
    
    def __ds_default(self):
        return ch.ArrayPlotData(t=self.t, 
                                y=self.data[:, 14],
                                yf=self.filter.df[:, 14])

    @tr.on_trait_change('filter.replot')                            
    def replot(self):
        self._ds.set_data('yf', self.filter.df[:, 14])        
        
    def __plot_default(self):        
        pl = ch.Plot(self._ds)
        pl.plot(('t', 'y'), color='black')        
        pl.plot(('t', 'yf'), color='red')        
        return pl
    
    plot_item = ui.Item('_plot', editor=en.ComponentEditor(), show_label=False)
    traits_view = ui.View(ui.VGroup([plot_item, '@filter']))

FilterTool(data=d[..., 0], t=range(len((t)))).edit_traits()