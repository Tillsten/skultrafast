import numpy as np
import time

from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'

# ETS imports (non-chaco)
from enable.component_editor import ComponentEditor
from enable.api import BaseTool
from traits.api import HasTraits, Instance, Int, List, Str, Enum, \
        on_trait_change, Any, DelegatesTo, Array,  Button, Float, Bool, Function

from traitsui.api import Item, View, HSplit, HGroup,\
        EnumEditor, Group, UItem, VGroup

# Chaco imports
from chaco.api import ArrayPlotData, Plot, ColorBar
import chaco.api as dc

from chaco.example_support import COLOR_PALETTE
from chaco.tools.api import PanTool, ZoomTool, RangeSelection, \
        RangeSelectionOverlay, LegendTool



from multiplotter import MultiPlotter, FitPlotter

from fitter import Fitter
import dv

class Data(HasTraits):
    wavelengths=Array()
    times=Array()
    data=Array(shape=(times.size,wavelengths.size))
    has_fit=Bool(default=False)
    fit_data=Array
    pd=Instance(ArrayPlotData)
    plot=Instance(Plot)

    spectrum_plotter=Instance(FitPlotter)
    transients_plotter=Instance(FitPlotter)
    reset_plotters=Button('Reset Plotters')

    v_container=Instance(dc.VPlotContainer)
    h_container=Instance(dc.HPlotContainer)

    def _spectrum_plotter_default(self):
        mp=FitPlotter(xaxis=self.wavelengths)
        mp.xlabel="nm"
        mp.ylabel="OD"
        return mp

    def _transients_plotter_default(self):
        mp=FitPlotter(xaxis=self.times)
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
        try:
            self.transients_plotter.ytemp=self.data[:,x]
            self.transients_plotter.plot.title=str(self.wavelengths[x])
            self.spectrum_plotter.ytemp=self.data[y,:]
            self.spectrum_plotter.plot.title=str(self.times[y])
        except IndexError:
            pass

    def add_transient(self,x,y):        
        y=self.data[:,x]
        if self.has_fit:
            y_fit=self.fit_data[:, x]
            
        name=str(round(self.wavelengths[x],2))
        self.transients_plotter.add_ydata(name,y, y_fit)

    def add_spectrum(self,x,y):
        yd=self.data[y,:]
        name=str(round(self.times[y],2))
        self.spectrum_plotter.add_ydata(name,yd)

    def _reset_plotters_fired(self):
        self.spectrum_plotter.delete_ydata()
        self.transients_plotter.delete_ydata()

    title='Data'
    size=(600,400)
    traits_view = View(
                    HGroup(
                            VGroup(Item('v_container', editor=ComponentEditor(size=size),
                                        show_label=False),
                                   Item('reset_plotters',show_label=False)),
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


from lmtraits import AllParameter


class FitterWindow(HasTraits):
    fitter=Instance(Fitter)
    para=Instance(AllParameter)
    dasplotter=Instance(MultiPlotter)
    resplotter=Instance(Plot)
    
    def _dasplotter_default(self):
        mp=MultiPlotter(xaxis=self.fitter.wl)
        mp.xlabel="nm"
        mp.ylabel="OD"
        return mp
        
    def _resplotter_default(self):
#        colorbar = ColorBar(index_mapper=dc.LinearMapper(range=colormap.range),
#                        color_mapper=colormap,
#                        orientation='v',
#                        resizable='v',
#                        width=30,
#                        padding=20)
#        colorbar.tools.append(RangeSelection(component=colorbar))
#        colorbar.overlays.append(RangeSelectionOverlay(component=colorbar,
#                                                   border_color="white",
#                                                   alpha=0.8,
#                                                   fill_color="lightgray"))
        res=np.zeros((400,400))
        self.pd=ArrayPlotData(res=res)
        p=Plot(self.pd)
        p.img_plot('res')
        return p
        
    def _para_default(self):
        para=AllParameter()
        return para

    @on_trait_change('para:paras:value')
    def calc_res(self):
        self.fitter.res(self.para.to_array())
        self.dasplotter.delete_ydata()
        tau=[str(round(i.value,2)) for i in self.para.paras[2:]]
        spec=[i for i in self.fitter.c[:-4]]
        for i in zip(tau,spec):
            self.dasplotter.add_ydata(*i)
        
        self.pd.set_data('res',self.fitter.m.T-self.fitter.data)

    @on_trait_change('para:start_fit')
    def start_fit(self):
        a=self.fitter.start_lmfit(self.para.to_array())
        a.params=self.para.to_lmparas()
        a.leastsq()
        self.para.from_lmparas(a.params)
        self.calc_res()

    vg=VGroup(Item('@para',show_label=False,height=300,width=0.7),
              Item('@resplotter', show_label=False, editor=ComponentEditor()))
    traits_view=View(HGroup(vg,
                            Item('@dasplotter',show_label=False,width=0.3)
                            ),resizable=True)






class ZeroCorrection(HasTraits):
    method=Enum(['Max', 'Abs', 'First Peak'],default='Abs')
    method_para=Float(default=2., label='Value')
    poly_degrees=Int(min=1,max=5, default=3)


class ZeroCorrectionTool(HasTraits):
    zc=Instance(ZeroCorrection)
    fitter=Instance(Fitter)
    plot=Instance(Plot)
   
    def _zc_default(self):
        return ZeroCorrection()
    
    def _plot_default(self):
        fitter=self.fitter
        idx=dv.find_linear_part(fitter.t)        
        self.pd=ArrayPlotData(wl=fitter.wl, t=fitter.t[:idx], d=fitter.data[:idx,:])
        plot=Plot(self.pd)
        plot.img_plot('d')
        plot.plot(('wl','tn'))        
        return plot
        
    size=((400,400))
    traits_view=View(Item('plot',editor=ComponentEditor(size=size),
                                  show_label=False), Item('@zc'))
        

class MainWindow(HasTraits):
    fitter=Instance(Fitter)
    data=Instance(Data)
    fitwin=Instance(FitterWindow)

    traits_view= View(VGroup(Item('@data',show_label=False, width=0.7),
                             Item('fitwin',width=0.3)
                             ),resizable=True
                      )

    def _data_default(self):
        t,w,d=self.fitter.t, self.fitter.wl, self.fitter.data
        data=Data(wavelengths=w,times=t,data=d)
        return data

    def _fitwin_default(self):
        return FitterWindow(fitter=self.fitter)

class FFTTool(HasTraits):
    x=Array
    y=Array
    Fs=Float

    data_plot=Any
    sel_range=Instance(RangeSelection)
    pd=Instance(ArrayPlotData)
    pd2=Instance(ArrayPlotData)
    freq_plot=Instance(Plot)
    v_container=Instance(dc.VPlotContainer)

    fft_window=Enum(('none','bartlett','blackman','hanning','kaiser'), default='hanning')
    func=Function(np.ones)

    detrend=Enum(('None','Mean','Linear'))    
    
    @on_trait_change('fft_window')
    def set_window_func(self):
        if self.fft_window!=None:
            self.func=eval('np.'+self.fft_window)
        else:
            self.func=lambda x: 1.

    def _Fs_default(self):
        return self.x[1]-self.x[0]


    def _pd_default(self):
        pd=ArrayPlotData(freqs=np.zeros(20),amps=np.zeros(20))
        return pd


    def _pd2_default(self):
        return ArrayPlotData(x=self.x, y=self.y)

    def _data_plot_default(self):
        plot=Plot(self.pd2)
        k=plot.plot(('x','y'))
        zoom = ZoomTool(component=plot, tool_mode="box", always_on=False)
        plot.overlays.append(zoom)
        
        k[0].active_tool = RangeSelection(k[0], left_button_selects = True)
        k[0].overlays.append(RangeSelectionOverlay(component=k[0]))
        self.sel_range=k[0].active_tool
        return plot

    @on_trait_change('sel_range:selection, fft_window')
    def calc_fft(self):       
        new=self.sel_range.selection        
        if new is None: return
        m=(self.x<new[1])&(self.x>new[0])
        n=np.sum(m)
        if sum(m)>3:
            d=self.y[m]
            win=self.func(n)
            y=np.abs(np.fft.fft(win*(d-d.mean())))
            x=np.fft.fftfreq(n, self.Fs)[:y.size/2]
            self.pd.set_data('freqs',x[1:])
            self.pd.set_data('amps',y[1:y.size/2])


    def _freq_plot_default(self):
        plot=Plot(self.pd)
        plot.plot(('freqs','amps'))
        return plot

    def _v_container_default(self):
        con=dc.VPlotContainer(self.data_plot, self.freq_plot)
        return con



    trait_view=View(VGroup(Item('data_plot', editor=ComponentEditor()),
                           Item('freq_plot', editor=ComponentEditor()),
                           Item('fft_window'),show_labels=False),resizable=True)
    #traits_view=View(Item('v_container', editor=ComponentEditor()))

from scipy import ndimage

#x=np.linspace(0,20,500)
#y=np.sin(x/1.2)
import dv

a=np.loadtxt('..\\al_tpfc2_ex620_magic.txt')

#t,wl,a=dv.loader_func('..\\messpy2\\tmp\\fremdprobe_para_400exec')
#wl, d=dv.concate_data(wl,a)
#t/=1000.

t=a[1:,0]#/1000.
w=a[0,1:]
d=a[1:,1:]

import scipy.ndimage as nd
#d=nd.uniform_filter(d,(3,5))
#f=FFTTool(x=w[:],y=d[4,:]) 
#f.configure_traits()

class f:
    pass
#d=d[...,:].mean(-1)
d=d[:,:]
wl=w[:]

a=f()
a.wl=wl
a.t=t


d=ndimage.gaussian_filter(d,(1,4))
a.data=d
#tn,p=dv.find_time_zero(a,3,polydeg=4)
#d=dv.interpol(d,t,tn,0.4,t)

f=Fitter(wl,t,d,1)
f.weights=np.median(d,0)
#dData(wavelengths=w,times=t,data=d)
m=MainWindow(fitter=f)
m.configure_traits()
