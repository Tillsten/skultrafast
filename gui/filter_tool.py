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
import scipy.ndimage as nd        
import skultrafast.dv as dv

class Filter(tr.HasTraits):
    t = tr.Array
    d = tr.Array
    df = tr.Array
    yf = tr.Array
    
    chan = tr.Int(0)
    replot = tr.Event
        
    def _df_default(self):        
        df = np.zeros_like(self.d)
        return df       
    
    def _yf_default(self):
        return self.d[:, 0]
        
    def _chan_changed(self, val):        
        self.update()        
        
    def update(self):
        self.replot = True
        
class UniformFilter(Filter):
    width = tr.Range(1, 10)                
    
    def _chan_changed(self):
        self.update()
    
    @tr.on_trait_change('width')
    def update(self):
        self.yf = nd.uniform_filter1d(self.d[:, self.chan], self.width, mode='nearest')
        self.replot = True
        
    traits_view = ui.View(['width'])
    
class SavGolFilter(Filter):
    order = tr.Range(1, 10)
    window_size = tr.Int(5)
    
    @tr.on_trait_change('order, window_size')
    def update(self):
        ws = (self.window_size / 2) * 2 + 1        
        y = self.d[:, self.chan]
        self.yf = dv.savitzky_golay(y, ws, self.order)
        self.replot = True
    
    range_edit = ui.RangeEditor(low_name='order', high=51)
    view_window_size = ui.Item('window_size', editor=range_edit)
    traits_view = ui.View(['order', view_window_size])
    
class GaussianFilter(Filter):
    width = tr.Range(1, 10)                
    
    def _chan_changed(self):
        self.update()
    
    @tr.on_trait_change('width')
    def update(self):
        self.yf = nd.gaussian_filter1d(self.d[:, self.chan], self.width, mode='nearest')
        self.replot = True
        
    traits_view = ui.View(['width'])

class SvdFilter(Filter):
    number_of_components = tr.Range(1, 20, 5)    
    
    def _df_default(self):
        self.apply_filter()
        return self.df
    
    @tr.on_trait_change('number_of_components')
    def apply_filter(self):                
        u, s, v = np.linalg.svd(self.d, full_matrices=False)
        s[self.number_of_components:] = 0
        s = np.diag(s)
        self.df = u.dot(np.dot(s, v))   
        self.yf = self.df[:, self.chan]
        self.replot = True
        return self.df
    
    def _chan_changed(self, n):
        self.yf = self.df[:, n]
        self.replot = True
          
    traits_view = ui.View(['number_of_components'])

filter_dict = {'SVD': SvdFilter,
               'Uniform': UniformFilter, 
               'Gaussian': GaussianFilter,
               'Savitzky-Golay': SavGolFilter}        

class FilterTool(tr.HasTraits):    
    data = tr.Array    
    max_chans = tr.Int    
    t = tr.Array     
    channel = tr.Int
    filter_type = tr.Enum(filter_dict.keys())
    filter = tr.Instance(Filter)
    _ds = tr.Instance(ch.ArrayPlotData)
    _plot = tr.Instance(ch.Plot)        

    def _max_chans_default(self):
        return int(self.data.shape[1]-1)

    def _filter_default(self):        
        return SvdFilter(t=self.t, d=self.data)
    
    def __ds_default(self):
        return ch.ArrayPlotData(t=self.t, 
                                y=self.data[:, self.channel],
                                yf=self.filter.yf)
        
    def __plot_default(self):        
        pl = ch.Plot(self._ds)
        pl.plot(('t', 'y'), color='black')        
        pl.plot(('t', 'yf'), color='red', line_width=1.2)        
        return pl

    @tr.on_trait_change('filter.replot')                            
    def replot(self):
        self._ds.set_data('yf', self.filter.yf)        
    
    def _channel_changed(self):
        self._ds.set_data('y', self.data[:, self.channel])        
        self.filter.chan = int(self.channel)
        self.replot()
    
    
    def _filter_type_changed(self, value):
        self.filter = filter_dict[value](t=self.t, d=self.data)
        
        
    plot_item = ui.Item('_plot',
                        editor=en.ComponentEditor(),
                        show_label=False)
    
    ch_item = ui.Item('channel', editor=ui.RangeEditor(low=0, 
                                                       high_name='max_chans',
                                                       is_float=False))
    settings_group =  ui.VGroup([ch_item, 'filter_type', '@filter'])
                        
    traits_view = ui.View(ui.HGroup([plot_item,  settings_group
                                    ]))

FilterTool(data=d[..., 0].T, t=range(len((w)))).edit_traits()