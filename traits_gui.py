
from numpy import linspace, random, zeros, arange, cumprod
import numpy as np
import time

from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'wx'

# ETS imports (non-chaco)
from enable.component_editor import ComponentEditor
from enable.api import BaseTool
from traits.api import HasTraits, Instance, Int, List, Str, Enum, \
        on_trait_change, Any, DelegatesTo, Array,  Button, Float, Bool
from traitsui.api import Item, View, HSplit, HGroup,\
        EnumEditor, Group, UItem, VGroup, TableEditor,\
        TabularEditor
from traitsui.table_column import ObjectColumn
# Chaco imports
from chaco.api import ArrayPlotData, Plot, PlotAxis, \
        ScatterInspectorOverlay, jet, GridDataSource, GridMapper, \
        DataRange2D, ImageData, CMapImagePlot, DataRange1D
import chaco.api as dc
from chaco.scales.api import CalendarScaleSystem
from chaco.scales_tick_generator import ScalesTickGenerator
from chaco.example_support import COLOR_PALETTE
from chaco.tools.api import PanTool, ZoomTool, RangeSelection, \
        RangeSelectionOverlay, LegendTool



from multiplotter import MultiPlotter 


          

class Data(HasTraits):
    wavelengths=Array()
    times=Array()
    data=Array(shape=(times.size,wavelengths.size))
    pd=Instance(ArrayPlotData)    
    plot=Instance(Plot)
    
    spectrum_plotter=Instance(MultiPlotter)
    transients_plotter=Instance(MultiPlotter)
    
    v_container=Instance(dc.VPlotContainer)
    h_container=Instance(dc.HPlotContainer)
    
    def _spectrum_plotter_default(self):
        mp=MultiPlotter(xaxis=self.wavelengths)
        mp.xlabel="nm"
        mp.ylabel="OD"
        return mp 
        
    def _transients_plotter_default(self):
        mp=MultiPlotter(xaxis=self.times)
        mp.xlabel="ps"
        mp.ylabel="OD"
        return mp 
        
    def _pd_default(self):
        pd = ArrayPlotData()
        pd.set_data("z", self.data)        
        return pd
        
    def _plot_default(self):        
        cmap_plot=Plot(self.pd)        
        cmap_plot.img_plot('z')                           
        
        cmap_plot.tools.append(ClickTool(self.hover_data,
                                          self.add_spectrum,
                                          self.add_transient,cmap_plot))
        #cmap_plot.tools.append(PanTool(cmap_plot))
        zoom = ZoomTool(component=cmap_plot, tool_mode="range", always_on=False)
        cmap_plot.overlays.append(zoom)
        return cmap_plot 
        
    def _v_container_default(self):
        con=dc.VPlotContainer(self.spectrum_plotter.plot, self.transients_plotter.plot)
        return con
    
    def _h_container_default(self):
        con=dc.HPlotContainer(self.plot,self.v_container)
        return con
    
    def hover_data(self,x,y):       
        self.transients_plotter.ytemp=self.data[:,x]
        self.transients_plotter.plot.title=str(self.wavelengths[x])
        self.spectrum_plotter.ytemp=self.data[y,:]
        self.spectrum_plotter.plot.title=str(self.times[y])
    
    def add_transient(self,x,y):
        y=self.data[:,x]
        name=str(round(self.wavelengths[x],2))
        self.transients_plotter.add_ydata(name,y)
        
    def add_spectrum(self,x,y):
        yd=self.data[y,:]
        name=str(round(self.times[y],2))
        self.spectrum_plotter.add_ydata(name,yd)
        
    def reset_plotter(self):        
        self.spectrum_plotter.delete_ydata()
        self.transients_plotter.delete_ydata()
        
    title='Data'
    size=(600,400)
    traits_view = View(
                    HGroup(Item('v_container', editor=ComponentEditor(size=size),
                             show_label=False),
                             Item('plot',editor=ComponentEditor(size=size),
                                  show_label=False)),
                             
                    resizable=True, title=title
                    )
                    

class ClickTool(BaseTool):
    def __init__(self, func, func_left, func_right, *args):
        super(ClickTool, self).__init__(*args)
        self.func=func
        self.func_left=func_left
        self.func_right=func_right

    def normal_mouse_move(self, event):        #
        x,y=self.component.map_data((event.x, event.y))
        self.func(x, y)
        
    def normal_left_down(self, event):
        x,y=self.component.map_data((event.x, event.y))
        self.func_left(x, y)
        
    def normal_right_down(self, event):
        x,y=self.component.map_data((event.x, event.y))
        self.func_right(x, y)

                    


class MainWindow(HasTraits):
    data=Instance(Data)
    do_calc=Button("Reset Plotters")
    
    traits_view= View(VGroup(UItem('@data'), Item('do_calc')))
    
    def _on_data_changed(self):
        self.traits_view=traits_view= View(VGroup(Item('@data'), Item('do_calc')))
    
    def _do_calc_fired(self):   
        self.data.reset_plotter()

#    
#a=np.loadtxt('..\\al_tpfc2_ex620_magic.txt')
#t=a[1:,0]
#w=a[0,1:]
#d=a[2:,2:]
#d=Data(wavelengths=w,times=t,data=d)
#m=MainWindow(data=d)
#m.configure_traits()