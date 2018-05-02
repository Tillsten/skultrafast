import sys
import matplotlib
#matplotlib.use('WxAgg') 
from matplotlib.widgets import MultiCursor, Cursor, Button
import matplotlib.pyplot as plt

matplotlib.rcParams['font.size'] = 8
import numpy as np





class fit_viewer(object):
    def __init__(self, global_fit_obj):
        self.gfo = global_fit_obj
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.subplots_adjust(0.05, 0.05, 0.98, 0.98, 0.1)
        self.image_data = self.gfo.data[:]
        self.overview = plt.subplot2grid((8, 4), (0, 0), rowspan=7, colspan=2)
        self.ov_plot = self.overview.pcolormesh(self.gfo.w, self.gfo.t, self.gfo.data)
        self.overview.autoscale(1, 'both', 1)
        self.spectrum = plt.subplot2grid((8, 4), (0, 2), rowspan=4, colspan=2)
        self.signal = plt.subplot2grid((8, 4), (4, 2), rowspan=4, colspan=2)
        self.mpl_connect = self.fig.canvas.mpl_connect
        self.cursor = Cursor(self.overview, color='black', linewidth=2)

        buax = plt.subplot2grid((8, 4), (7, 0))
        bu2ax = plt.subplot2grid((8, 4), (7, 1))
        self.reset_button = Button(buax, 'Reset')
        self.norm_button = Button(bu2ax, 'Change Norm')
        self.norm = 'no_norm'
        self.connect()

    def connect(self):
        self.fig.canvas.mpl_connect('button_press_event', self.clickax)
        self.reset_button.on_clicked(self.reset)
        self.norm_button.on_clicked(self.change_norm)

    def reset(self, event):
        for j in [self.overview, self.spectrum, self.signal]:
            j.lines = []
        plt.draw()

    def change_norm(self, event):
        if self.norm == 'no_norm':
            self.image_data = self.gfo.data[:] / (np.abs(self.gfo.data[-1, :]))
            self.overview.collections = []
            self.ov_plot = self.overview.pcolormesh(self.gfo.w, self.gfo.t, self.image_data, vmax=2, vmin=-2)

            self.norm = 'abs_norm'
        elif self.norm == 'abs_norm':
            self.image_data = self.gfo.data[:]
            self.overview.collections = []
            self.ov_plot = self.overview.pcolormesh(self.gfo.w, self.gfo.t, self.image_data)
            self.norm = 'no_norm'
        self.overview.autoscale(1, 'both', 1)
        plt.draw()

    def clickax(self, event):
        if event.inaxes == self.overview:
            wpos = np.argmin(np.abs(event.xdata - self.gfo.w))
            tpos = np.argmin(np.abs(event.ydata - self.gfo.t))
            if event.button == 3:
                c, = self.spectrum.plot(self.gfo.w, self.gfo.data[tpos, :], label=str(self.gfo.t[tpos]))
                #self.spectrum.plot(self.gfo.w, self.gfo.m.T[tpos,:],'k--',lw=1)
                self.overview.axhline(self.gfo.t[tpos], color=c.get_color(), linewidth=2)
            elif event.button == 1:
                c, = self.signal.plot(self.gfo.t, self.gfo.data[:, wpos], label=str(self.gfo.w[wpos]))
                #c,=self.signal.plot(self.gfo.t, self.gfo.m.T[:,wpos],'k--',lw=1)            
                self.overview.axvline(self.gfo.w[wpos], color=c.get_color(), linewidth=2)
        if event.inaxes == self.signal:
            tpos = np.argmin(np.abs(event.xdata - self.gfo.t))
            c, = self.spectrum.plot(self.gfo.w, self.gfo.data[tpos, :], label=str(self.gfo.t[tpos]))
            self.overview.axhline(self.gfo.t[tpos], color=c.get_color(), linewidth=2)

        if event.inaxes == self.spectrum:
            wpos = np.argmin(np.abs(event.xdata - self.gfo.w))
            c, = self.signal.plot(self.gfo.t, self.gfo.data[:, wpos], label=str(self.gfo.w[wpos]))
            self.overview.axvline(self.gfo.w[wpos], color=c.get_color(), linewidth=2)

        plt.draw()


class g:
    pass


def make_view(name):
    t, wl, a = loader_func(name)
    wl, dat = concate_data(wl, a)
    c = g()
    c.w, c.t, c.data = wl, t, dat[..., :].mean(-1)
    return fit_viewer(c)


c = g()
ux = np.where((u>0.4)^(u<-.2), np.nan, u)
c.w, c.t, c.data = w, t, ux
fit_viewer(c)
#dat=np.load('tmp\\br_py2_ex_590_senk-525_0_dat.npy')
#
#w,t=dat[0,1:], dat[1:,0]
#dats=dat[1:,1:]
#dat=np.load('tmp\\br_py2_ex_590_senk-660_0_dat.npy')
#w2=dat[0,1:]
#w=hstack((w,w2))
##dat=np.load('also_638_dat5.npy').squeeze()
##std=np.load('also_638_std5.npy').squeeze()
#dat=np.load('tmp\\br_py2_ex_590_parallel_dat10.npy').squeeze()
#std=np.load('tmp\\br_all_wl2_std20.npy').squeeze()
#
#dats=-np.average(dat,-1)
#
#dats=np.hstack((dats[:,:,0],dats[:,:,1]))
#idx=np.argsort(w)
#w.sort()
#dats=dats[:,idx]
#dats-=dats[:5,:].mean(0)from dv import loader_func
#G=g()
#G.w=w
#G.t=t
#G.data=-dats
#
#fig_v=fit_viewer(G)
#
#dat=np.load('tmp\\br_py2_ex_590_senk_dat8.npy').squeeze()
#std=np.load('tmp\\br_all_wl2_std20.npy').squeeze()
#
#dats=-np.average(dat,-1)
#dats=np.hstack((dats[:,:,0],dats[:,:,1]))
#dats-=dats[:5,:].mean(0)
#dats=dats[:,idx]
#f=g()
#f.w=w
#f.t=t
#f.data=-dats
#fig_v=fit_viewer(f)
#
#plt.show()
#
#
#plt.show()

