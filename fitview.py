import sys
import matplotlib
#matplotlib.use('WxAgg') 
from matplotlib.widgets import MultiCursor, Cursor, Button
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size']=8
import numpy as np
import cPickle
import dv
from dv import loader_func
from dv import concate_data
    


class fit_viewer(object):
    def __init__(self,global_fit_obj):        
        self.gfo=global_fit_obj
        self.fig=plt.figure(figsize=(12,10))
        self.fig.subplots_adjust(0.05,0.05,0.98,0.98,0.1)
        self.image_data=self.gfo.data[:]
        self.overview=plt.subplot2grid((8,4),(0,0),rowspan=7,colspan=2)
        self.ov_plot=self.overview.pcolormesh(self.gfo.w,self.gfo.t,self.gfo.data)
        self.overview.autoscale(1,'both',1)
        self.spectrum=plt.subplot2grid((8,4),(0,2),rowspan=4,colspan=2)
        self.signal=plt.subplot2grid((8,4),(4,2),rowspan=4,colspan=2)
        self.mpl_connect=self.fig.canvas.mpl_connect
        self.cursor = Cursor(self.overview, color='black', linewidth=2 )
        
        buax=plt.subplot2grid((8,4),(7,0))
        bu2ax=plt.subplot2grid((8,4),(7,1))
        self.reset_button=Button(buax,'Reset')
        self.norm_button=Button(bu2ax,'Change Norm')
        self.norm='no_norm'
        self.connect()
    
    def connect(self):
        self.fig.canvas.mpl_connect('button_press_event',self.clickax)       
        self.reset_button.on_clicked(self.reset)
        self.norm_button.on_clicked(self.change_norm)

    def reset(self,event):
        for j in [self.overview,self.spectrum,self.signal]:
            j.lines=[]
        plt.draw()
        
    def change_norm(self,event):
        if self.norm=='no_norm':           
            self.image_data=np.abs(self.gfo.data[:]/(np.abs(self.gfo.data[-10:,:].mean(0)+1.)))
            self.overview.collections=[]
            self.ov_plot=self.overview.pcolormesh(self.gfo.w,self.gfo.t,
                                                  self.image_data,vmax=1.5,cmap=plt.cm.Paired)
            
            self.norm='abs_norm'
        elif self.norm=='abs_norm':                      
            self.image_data=self.gfo.data[:]
            self.overview.collections=[]
            self.ov_plot=self.overview.pcolormesh(self.gfo.w,self.gfo.t,self.image_data,cmap=plt.cm.Paired)
            self.norm='no_norm'
        self.overview.autoscale(1,'both',1)
        plt.draw()
        
    def clickax(self,event):        
        if event.inaxes==self.overview:
            wpos=np.argmin(np.abs(event.xdata-self.gfo.w))           
            tpos=np.argmin(np.abs(event.ydata-self.gfo.t))            
            if event.button==3:            
                c,=self.spectrum.plot(self.gfo.w, self.gfo.data[tpos,:],label=str(self.gfo.t[tpos]))
                #self.spectrum.plot(self.gfo.w, self.gfo.m.T[tpos,:],'k--',lw=1)
                self.overview.axhline(self.gfo.t[tpos],color=c.get_color(),linewidth=2)
            elif event.button==1: 
                c,=self.signal.plot(self.gfo.t, self.gfo.data[:,wpos],label=str(self.gfo.w[wpos]))            
                #c,=self.signal.plot(self.gfo.t, self.gfo.m.T[:,wpos],'k--',lw=1)            
                self.overview.axvline(self.gfo.w[wpos],color=c.get_color(),linewidth=2)
        if event.inaxes==self.signal:
            tpos=np.argmin(np.abs(event.xdata-self.gfo.t))            
            c,=self.spectrum.plot(self.gfo.w, self.gfo.data[tpos,:],label=str(self.gfo.t[tpos]))
            self.overview.axhline(self.gfo.t[tpos],color=c.get_color(),linewidth=2)
        
        if event.inaxes==self.spectrum:
            wpos=np.argmin(np.abs(event.xdata-self.gfo.w))     
            c,=self.signal.plot(self.gfo.t, self.gfo.data[:,wpos],label=str(self.gfo.w[wpos]))    
            self.overview.axvline(self.gfo.w[wpos],color=c.get_color(),linewidth=2)
            
        plt.draw()

class g:
    pass

def argtake(arr, ind, axis=-1) :
    """Take using output of argsort
    *Description*

    The usual output of argsort is not easily used with take when the relevent
    array is multidimensional. This is a quick and dirty standin.

    *Parameters*:

        arr : ndarray
            The array from which the elements are taken. Typically this will be
            the same array to which argsort was applied.

        ind : ndarray
            Indices returned by argsort or lexsort.

        axis : integer
            Axis to which argsort or lexsort was applied. Must match the
            original call. Defaults to -1.

    *Returns*:

        reordered_array : same type as arr
            The input array reordered by ind along the axis.

    """
    N=np
    if arr.shape != ind.shape :
        raise "shape mismatch"
    if arr.ndim == 1 :
        return N.take(arr, ind)
    else :
        naxis = arr.shape[axis]
        aswap = N.swapaxes(arr, axis, -1)
        iswap = N.swapaxes(ind, axis, -1)
        shape = aswap.shape
        aswap = aswap.reshape(-1, naxis)
        iswap = iswap.reshape(-1, naxis)
        oswap = N.empty_like(aswap)
        for i in xrange(len(aswap)) :
            N.take(aswap[i], iswap[i], out=oswap[i])
        oswap = oswap.reshape(shape)
        oswap = N.swapaxes(oswap, axis, -1)
        return oswap

def make_view(name):
    from scipy.stats import nanmean
    t,wl,a=loader_func(name)
    plt.spectral()
    wl, dat=concate_data(wl,a)
    dat=dat[:,:,:2]
    dat=np.sort(dat)
    dat-=dat[:5,...].mean(0)
    c=g()
    #dat=dv.subtract_background(dat.mean(2),5)
    #tn=dv.poly_time_abs(dat,t,2,up=4000,w=wl)
    #dat=dv.interpol(dat,t,tn,0,t)
    print dat.shape, wl.shape, t.shape
    #i=np.argsort(t)
    #dat=dat[i,...]
    #t=t[i]
    j=dat[:,:,:].mean(-1)
    m=np.abs(dat-dat.mean(-1)[:,:,None])
    print m.shape
    k=m>10*np.std(dat,-1)[:,:,None]
    print k.sum()
    udat=dat.copy()
    dat[k]=np.nan        
    import bottleneck as bn
    #j=nanmean(dat,-1)
    
    #j=bn.move_mean(j,4,0)
    c.w, c.t, c.data=wl, t[:], j[:]
    #fit_viewer(c)
    #d=g()
    #d.w, d.t, d.data=wl, t, np.mean(udat,-1)
    return fit_viewer(c), t,wl,dat, a
    



def make_view(name):
    from scipy.stats import nanmean
    t,wl1,a, b=loader_func(name)
    wl, dat=concate_data(wl1,a)
    wl, std=concate_data(wl1,b)
    dat=dat[:,:,:]
    dat-=dat[:6,...].mean(0)
    i=np.argsort(np.abs(dat-np.median(dat,-1)[...,None]),axis=-1)
    #print i.shape, dat.shape
    dat=argtake(dat,i)
    
    c=g()
    c.w, c.t, c.data=wl, t[:], np.average(dat[...,:10],-1)#/(std[...,:9])**2)#dat[:,:,:6].mean(-1)
    return fit_viewer(c), t,wl,dat, a, b
#V, t, wl, dat, a=make_view('tmp\\5th_al_py2_ex620_magic')
V, t, wl, dat, a, b=make_view('tmp\\7_al_py2_ex620_magic')
#V2, t, wl, dat2=make_view('tmp\\br_py_ex640_para')
#



        
from pylab import *
ion()
plt.show()

