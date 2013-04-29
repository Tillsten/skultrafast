# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 21:54:56 2012

@author: Tillsten
"""

from traits.api import HasTraits, Instance, Int, List, Str, Enum,\
    on_trait_change, Any, DelegatesTo, Array, Button, Float, Bool, Function

from traitsui.api import View, Group, Item, VGroup

from enable.component_editor import ComponentEditor

from chaco.api import ArrayPlotData, Plot, VPlotContainer
from chaco.tools.api import PanTool, ZoomTool, RangeSelection,\
    RangeSelectionOverlay, LegendTool

import numpy as np

class FFTTool(HasTraits):
    x = Array
    y = Array
    Fs = Float

    data_plot = Any
    sel_range = Instance(RangeSelection)
    pd = Instance(ArrayPlotData)
    pd2 = Instance(ArrayPlotData)
    
    freq_plot = Instance(Plot)
    v_container = Instance(VPlotContainer)

    fft_window = Enum(('none', 'bartlett', 'blackman', 'hanning', 'kaiser'), default='hanning')
    func = Function(np.ones)

    detrend = Enum(('None', 'Mean', 'Linear'))

    @on_trait_change('fft_window')
    def set_window_func(self):
        if self.fft_window != None:
            self.func = eval('np.' + self.fft_window)
        else:
            self.func = lambda x: 1.

    def _Fs_default(self):
        return self.x[1] - self.x[0]


    def _pd_default(self):
        pd = ArrayPlotData(freqs=np.zeros(20), amps=np.zeros(20))
        return pd

    def _pd2_default(self):
        return ArrayPlotData(x=self.x, y=self.y, x_sel=[], y_cor=[])
        
    

    def _data_plot_default(self):
        plot = Plot(self.pd2)
        plot.plot(('x_sel', 'y_cor'))
        k = plot.plot(('x', 'y'))
        zoom = ZoomTool(component=plot, tool_mode="box", always_on=False)
        plot.overlays.append(zoom)

        k[0].active_tool = RangeSelection(k[0], left_button_selects=True)
        k[0].overlays.append(RangeSelectionOverlay(component=k[0]))
        self.sel_range = k[0].active_tool
        return plot

    @on_trait_change('sel_range:selection, fft_window')
    def calc_fft(self):
        new = self.sel_range.selection
        if new is None: return
        m = (self.x < new[1]) & (self.x > new[0])
        n = np.sum(m)
        if sum(m) > 3:
            d = self.y[m]
            xd = self.x[m]
            win = self.func(n)
            back = np.poly1d(np.polyfit(xd, d, 3))(xd)     
            self.pd2.set_data('x_sel', xd)            
            y = np.abs(np.fft.fft(win * (d - back)))
            self.pd2.set_data('y_cor', win * (d - back))
            x = np.fft.fftfreq(n, self.Fs)[:y.size / 2]
            x = dv.fs2cm(1000./x)
            self.pd.set_data('freqs', x[1:])
            self.pd.set_data('amps', y[1:y.size / 2])


    def _freq_plot_default(self):
        plot = Plot(self.pd)
        plot.plot(('freqs', 'amps'))
        return plot

    def _v_container_default(self):
        con = VPlotContainer(self.data_plot, self.freq_plot)
        return con


    trait_view = View(VGroup(Item('data_plot', editor=ComponentEditor()),
        Item('freq_plot', editor=ComponentEditor()),
        Item('fft_window'), show_labels=False), resizable=True)

#print dv.fi(w, 650)
#FFTTool(x=t[104:], y=c).configure_traits()