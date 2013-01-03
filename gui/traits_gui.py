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
    EnumEditor, Group, UItem, VGroup

# Chaco imports
from chaco.api import ArrayPlotData, Plot, ColorBar, HPlotContainer
import chaco.api as dc

from chaco.example_support import COLOR_PALETTE
from chaco.tools.api import PanTool, ZoomTool, RangeSelection,\
    RangeSelectionOverlay, LegendTool

from multiplotter import MultiPlotter, FitPlotter

from fitter import Fitter
import dv

class Data(HasTraits):
    wavelengths = Array()
    times = Array()
    data = Array(shape=(times.size, wavelengths.size))
    has_fit = Bool(default=False)
    fit_data = Array
    pd = Instance(ArrayPlotData)
    plot = Instance(Plot)

    spectrum_plotter = Instance(FitPlotter)
    transients_plotter = Instance(FitPlotter)
    reset_plotters = Button('Reset Plotters')

    v_container = Instance(dc.VPlotContainer)
    h_container = Instance(dc.HPlotContainer)

    def _spectrum_plotter_default(self):
        mp = FitPlotter(xaxis=self.wavelengths)
        mp.xlabel = "nm"
        mp.ylabel = "OD"
        return mp

    def _transients_plotter_default(self):
        mp = FitPlotter(xaxis=self.times)
        mp.xlabel = "ps"
        mp.ylabel = "OD"
        return mp

    def _pd_default(self):
        pd = ArrayPlotData()
        pd.set_data("z", self.data)
        return pd

    def _plot_default(self):
        cmap_plot = Plot(self.pd)
        cmap_plot.img_plot('z')

        cmap_plot.tools.append(ClickTool(self.hover_data,
            self.add_spectrum,
            self.add_transient, cmap_plot))
        #cmap_plot.tools.append(PanTool(cmap_plot))
        zoom = ZoomTool(component=cmap_plot, tool_mode="range", always_on=False)
        cmap_plot.overlays.append(zoom)
        return cmap_plot

    def _v_container_default(self):
        con = dc.VPlotContainer(self.spectrum_plotter.plot, self.transients_plotter.plot)
        return con

    def _h_container_default(self):
        con = dc.HPlotContainer(self.plot, self.v_container)
        return con

    def hover_data(self, x, y):
        try:
            self.transients_plotter.ytemp = self.data[:, x]
            self.transients_plotter.plot.title = str(self.wavelengths[x])
            self.spectrum_plotter.ytemp = self.data[y, :]
            self.spectrum_plotter.plot.title = str(self.times[y])
        except IndexError:
            pass

    def add_transient(self, x, y):
        name = str(round(self.wavelengths[x], 2))
        y = self.data[:, x]
        if self.has_fit:
            y_fit = self.fit_data[:, x]
            self.transients_plotter.add_ydata(name, y, y_fit)
        else:
            self.transients_plotter.add_ydata(name, y)


    def add_spectrum(self, x, y):
        name = str(round(self.times[y], 2))
        y = self.data[y, :]
        if self.has_fit:
            y_fit = self.fit_data[x, :]
            self.spectrum_plotter.add_ydata(name, y, y_fit)
        else:
            self.spectrum_plotter.add_ydata(name, y)


    def _reset_plotters_fired(self):
        self.spectrum_plotter.delete_ydata()
        self.transients_plotter.delete_ydata()

    title = 'Data'
    size = (600, 400)
    traits_view = View(
        HGroup(
            VGroup(Item('v_container', editor=ComponentEditor(size=size),
                show_label=False),
                Item('reset_plotters', show_label=False)),
            Item('plot', editor=ComponentEditor(size=size),
                show_label=False)),

        resizable=True, title=title
    )


class ClickTool(BaseTool):
    """
    Simple chaco tool to handle clicks and moving of the cursor on
    chaco plots.
    """
    def __init__(self, func_move, func_left, func_right, *args):
        super(ClickTool, self).__init__(*args)
        self.func_move = func_move
        self.func_left = func_left
        self.func_right = func_right

    def normal_mouse_move(self, event):        #
        x, y = self.component.map_data((event.x, event.y))
        self.func_move(x, y)

    def normal_left_down(self, event):
        x, y = self.component.map_data((event.x, event.y))
        self.func_left(x, y)

    def normal_right_down(self, event):
        x, y = self.component.map_data((event.x, event.y))
        self.func_right(x, y)


from lmtraits import AllParameter

class FitInfo(HasTraits):
    chi_sq = Float
    
class FitterWindow(HasTraits):
    fitter = Instance(Fitter)
    para = Instance(AllParameter)
    dasplotter = Instance(MultiPlotter)
    resplotter = Instance(HPlotContainer)

    def _dasplotter_default(self):
        mp = MultiPlotter(xaxis=self.fitter.wl)
        mp.xlabel = "nm"
        mp.ylabel = "OD"
        return mp

    def _resplotter_default(self):
        res = np.zeros((400, 400))
        self.pd = ArrayPlotData(res=res)
        p = Plot(self.pd)
        img = p.img_plot('res', name="my_plot")
        my_plot = p.plots["my_plot"][0]
        colormap = my_plot.color_mapper
        colorbar = ColorBar(
            index_mapper=dc.LinearMapper(range=colormap.range),
            color_mapper=colormap,
            orientation='v',
            plot=my_plot,
            resizable='v',
            width=30,
            padding=20)

        range_selection = RangeSelection(component=colorbar)
        colorbar.tools.append(range_selection)
        colorbar.overlays.append(RangeSelectionOverlay(component=colorbar,
            border_color="white",
            alpha=0.8,
            fill_color="lightgray"))
        range_selection.listeners.append(my_plot)
        con = HPlotContainer()
        con.add(p)
        con.add(colorbar)
        return con

    def _para_default(self):
        para = AllParameter()
        return para

    @on_trait_change('para:paras:value')
    def calc_res(self):
        self.fitter.res(self.para.to_array())
        self.dasplotter.delete_ydata()
        tau = [str(round(i.value, 2)) for i in self.para.paras[2:]]
        spec = [i for i in self.fitter.c[:-4]]
        for i in zip(tau, spec):
            self.dasplotter.add_ydata(*i)
        self.pd.set_data('res', self.fitter.m.T - self.fitter.data)

    @on_trait_change('para:start_fit')
    def start_fit(self):
        a = self.fitter.start_lmfit(self.para.to_array())
        a.params = self.para.to_lmparas()
        a.prepare_fit(a.params)
        a.leastsq()
        self.para.from_lmparas(a.params)
        self.calc_res()

    vg = VGroup(
        Item('@para', show_label=False, height=300, width=0.7),
        Item('@resplotter', show_label=False, editor=ComponentEditor()),
        )
    traits_view = View(
        HGroup(vg,
           Item('@dasplotter', show_label=False, width=0.3)),
        resizable=True)



class MainWindow(HasTraits):
    fitter = Instance(Fitter)
    data = Instance(Data)
    data_fit = Array
    fitwin = Instance(FitterWindow)

    traits_view = View(VGroup(Item('@data', show_label=False, width=0.7),
        Item('fitwin', width=0.3)
    ), resizable=True
    )

    def _data_default(self):
        t, w, d = self.fitter.t, self.fitter.wl, self.fitter.data
        data = Data(wavelengths=w, times=t, data=d)
        return data

    def _data_fit_default(self):
        return self.fitter.m.T


    def _fitwin_default(self):
        return FitterWindow(fitter=self.fitter)


    @on_trait_change('fitwin:para:paras:value')
    def _data_fit_changed(self):
        self.data.has_fit = True
        self.data.fit_data = self.fitter.m.T


        #traits_view=View(Item('v_container', editor=ComponentEditor()))
if __name__=='__main__':
    from scipy import ndimage
    
    #x=np.linspace(0,20,500)
    #y=np.sin(x/1.2)
    import dv
    
    a = np.loadtxt('..\\alcor_py2_ex400.txt')
    
    #t,wl,a=dv.loader_func('..\\messpy2\\tmp\\fremdprobe_para_400exec')
    #wl, d=dv.concate_data(wl,a)
    #t/=1000.
    
    t = a[1:200, 0]#/1000.
    w = a[0, 1:]
    d = a[1:200, 1:]
    
    import scipy.ndimage as nd
    #d=nd.uniform_filter(d,(3,5))
    #f=FFTTool(x=w[:],y=d[4,:]) 
    #f.configure_traits()
    
    class f:
        pass
    
    #d=d[...,:].mean(-1)
    d = d[:, :]
    wl = w[:]
    
    a = f()
    a.wl = wl
    a.t = t[:]
    
    
    #d=ndimage.gaussian_filter(d,(1,4))
    a.data = d
    #tn,p=dv.find_time_zero(a,3,polydeg=4)
    #d=dv.interpol(d,t,tn,0.4,t)
    
    f = Fitter(wl, t, d, 1)
    f.weights = np.median(d, 0)
    #dData(wavelengths=w,times=t,data=d)
    m = MainWindow(fitter=f)
    m.configure_traits()
