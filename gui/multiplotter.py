# -*- coding: utf-8 -*-
"""
Created on Sat May 19 02:26:29 2012

@author: Tillsten
"""

from numpy import linspace, random, zeros, arange, cumprod
import numpy as np
import time


# ETS imports (non-chaco)
from enable.component_editor import ComponentEditor
from enable.api import BaseTool
from traits.api import HasTraits, Instance, Int, List, Str, Array, Bool
from traitsui.api import Item, View

# Chaco imports
from chaco.api import ArrayPlotData, Plot
from chaco.tools.api import PanTool, ZoomTool, RangeSelection, \
        RangeSelectionOverlay, LegendTool



from chaco.example_support import COLOR_PALETTE

#from enthought.chaco.example_support import COLOR_PALETTE
COLOR_PALETTE=['blue','green','red','cyan','yellow', 'orange','tomato','pink']

class MultiPlotter(HasTraits):
    plot=Instance(Plot)
    xaxis=Array
    ytemp=Array
    ydata=List(Array)
    pd=Instance(ArrayPlotData)
   
    xlabel=Str('')
    ylabel=Str('')    
    
    def _pd_default(self):
        pd=ArrayPlotData(x=self.xaxis,ytemp=self.ytemp)        
        return pd
        
    def _xlabel_changed(self,new):        
        self.plot.x_axis.title=new
        
    def _ylabel_changed(self,new):        
        self.plot.y_axis.title=new
     
    
    def _plot_default(self):
        plot=Plot(self.pd)          
        plot.plot(('x','ytemp'),name='cursor')
        plot.legend.visible=True
        plot.tools.append(PanTool(plot, drag_button='right'))        
        zoom = ZoomTool(component=plot, tool_mode="box", always_on=True)
        plot.overlays.append(zoom)          
        return plot
    
    def _ytemp_changed(self):
        self.pd.set_data('ytemp',self.ytemp)   

    def delete_ydata(self):
        self.ydata=[]        
        l=[]
        
        for i in self.plot.plots:
           l.append(i)
        print l
        for i in l:
            self.plot.delplot(i)
        self.plot.plot(('x','ytemp'),name='cursor')
        
    def add_ydata(self,name, y, lw=3, c=None):
        if name in self.plot.plots: return
        self.ydata.append(y)
        self.pd.set_data(name,y)       
        if c is None:
            c=COLOR_PALETTE[(len(self.ydata)-1)%len(COLOR_PALETTE)]
        
        self.plot.plot(('x',name),name=name,color=c,line_width=lw)
        self.plot.request_redraw()

        

    traits_view=View(Item('plot',editor=ComponentEditor(size=(400,400))))
    

class FitPlotter(MultiPlotter):
        plot_fit=Bool(default=False)
        ytemp_fit=Array
        
        def __init__(self,**kwargs):
            MultiPlotter.__init__(self,**kwargs)
            self.pd
            
            
        def add_ydata(self,name, y, yfit=None):
            self.ydata.append(y)
            self.pd.set_data(name,y)
            c=COLOR_PALETTE[(len(self.ydata)-1)%len(COLOR_PALETTE)]
            if not yfit==None:
                self.pd.set_data(name + '_fit', yfit)
                self.plot.plot(('x',name), type='scatter', name=name, color=c)
                self.plot.plot(('x',name+'_fit'), name=name+'_fit', color=c, line_width=3.)
            else:
                self.plot.plot(('x',name),name=name,color=c,line_width=3.)
            self.plot.request_redraw()
            


if __name__=='__main__':
    x=np.linspace(0,10,100)
    y=np.sin(x)         
    mlp=FitPlotter(xaxis=x)
    mlp.add_ydata('sin',y+np.random.randn(y.size), y)
    
    mlp.xlabel='dsfj'    
    #mlp.delete_ydata()
    mlp.configure_traits()
    
  