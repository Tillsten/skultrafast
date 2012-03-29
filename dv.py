# -*- coding: utf-8 -*-
from numpy import zeros, loadtxt, ceil, interp, around, arange, floor,log10
import numpy as np
from scipy.interpolate import splrep, splev
def testload(f = 'C:\Users\Tillsten\Documents\weisslicht.dat'):
    
    raw = load_datfile(f)
    (times, wl, sig, se, ref, re) = read_data(raw)  
    return  (times, wl, sig, se, ref, re)
    
def load_datfile(datfile):
    # Lade Daten aus vb-Datenfile, f konvertiert die Kommas
    f = lambda  s: float(s.replace(',', '.'))
    d = loadtxt(datfile, converters = {0:f, 1:f, 2:f, 3:f})
    return d



def find_w(w,x):
    """needs global w as array. gives idnex to nearest value"""    
    idx = (abs(w - x)).argmin()
    return idx
    
def read_data(d):    
    # Setze Ausgabenarrays und fuelle sie
    num_times = len(d) / (800)
    times = zeros((num_times))
    wl = zeros((400))#
    for i in range(0, 400):
        wl[i] = d[i, 1]
    sig = zeros((num_times, 400))
    ref = zeros((num_times, 400))
    sig_err = zeros((num_times, 400))
    ref_err = zeros((num_times, 400))
    for i in range(num_times):
        times[i] = d[i * 800 + 1, 0]
        for j in range(0, 400):
            sig[i, j] = d[i * 800 + j, 2]
            sig_err[i, j] = d[i * 800 + j, 3]
            ref[i, j] = d[i * 800 + j + 400, 2]
            ref_err[i, j] = d[i * 800 + j + 400, 3]
    return (times, wl, sig, sig_err, ref, ref_err)

def calc_od(sig):
    return - 1000 * log10(sig)

def first_min(sig, low_lim):
    i = 4
    while (not sig[i - 1] > sig[i] < sig[i + 1]) or sig[i] > low_lim:
        i += 1
        
        if i > sig.shape[0]-2:
            break
    return i
    
def loader_func(name):
    import glob
    

    files=glob.glob(name+'*dat?.npy')+glob.glob(name+'*dat??.npy')
    if len(files)==0:
        raise "Name Not Found"
    num_list=[i[i.find('dat')+3:-4] for i in files]
    #print num_list
    endname=max(zip(map(int,num_list),files))[1]
    print 'Loading: '+endname
    a=np.load(endname)
    files=glob.glob(name+'*'+'_0_'+'*dat.npy')
    #print files
    wls=[]
    for i in files:   
        print 'Loading: '+i
        tmp=np.load(i)
        t,w=tmp[1:,0],tmp[0,1:]
        wls.append(w)
    
    return t,wls,a
    
def concate_data(wls,dat):
    w=np.hstack(tuple(wls))
    idx=np.argsort(w)
    w=w[idx]
    k=dat.shape
    dat=dat.reshape(k[0],k[1]*k[2],k[3],order='F')
    dat=dat[:,idx,:]
    return w,dat
    


def get_abs(s, lim):
    z = zeros(s.shape[1], dtype = 'int')
    for i in range(s.shape[1]):
        j = 0
        while j < s.shape[0] - 2 and (s[j, i]) < lim:
            j = j + 1
        if j != s.shape[0] - 1: z[i] = j
    return z
    
def get_zeros(s, low_lim = -0.5):
    z = zeros(s.shape[1], dtype = 'int')
    for i in range(s.shape[1]):        
        z[i] = first_min(s[:, i], low_lim)
    return z
    
    
def subtract_background(sig, k = 10):
    c = np.mean(sig[0:k, :], axis = 0)    
    return sig - np.resize(c, sig.shape)

def process(name):
    (t, w, s, se, r, re) = testload(name)
    r = calc_od(r)
    r = subtract_background(r)
    
    return t, w, r

def poly_time(k, t, level = 0.25, up = 2000, low = -500,w=np.arange(400)):    
    k = t[get_zeros(k, level)]
    m = np.ma.masked_array(k, np.logical_or(k<low, k>up))    
    masked_range = np.ma.masked_array(w, m.mask)
    p = np.polyfit(masked_range.compressed(), m.compressed(), 2)
    return(np.poly1d(p)(np.arange(k.shape[0])))
    
def poly_time_abs(k, t, level = -1, up = 2000, low = -1000,w=arange(400)):    
    k = t[np.argmin(abs(k)<level,0)]
    m = np.ma.masked_array(k, np.logical_or(k<low, k>up))    
    masked_range = np.ma.masked_array(w, m.mask)
    p = np.polyfit(masked_range.compressed(), m.compressed(), 3)
    print p
    return(np.poly1d(p)(w))

def poly_time_abs_r(k, t,w, level = -1, up = 2000, low = -1000):    
    k = t[get_abs(k, level)]
    m = np.ma.masked_array(k, np.logical_or(k<low, k>up))    
    masked_range = np.ma.masked_array(w, m.mask)
    p = np.polyfit(masked_range.compressed(), m.compressed(), 3)
    print p
    return(np.poly1d(p)(arange(k.shape[0])))

def spec_plt(g,plt):
    plt.xlabel(u'Wellenlänge  /  nm')
    plt.ylabel(u'Absorptionsdifferenz / mOD')
    plt.gcf().set_figheight(4.5)
    plt.gcf().set_figwidth(7)
    grid()
    para=g.a.T[:g.a[0].T.size/2]
    senk=g.a.T[g.a[0].T.size/2:]
    plt.plot(g.wl,para,lw=2)
    plt.plot(g.wl,senk)
    
def interpol(r,t,time_zero,shift,new_t=np.array([-500,-250,0,250,500,1000,1500,2000,500000])):    
    t_array=np.tile(t.reshape(t.size,1),(1,r.shape[1]))
    r_new=zeros((new_t.size,r.shape[1]))
    for i in range(r.shape[1]):
        t_array[:,i]-=time_zero[i]-shift
        r_new[:,i]=interp(new_t,t_array[:,i],r[:,i])
    return r_new

def interpol_sm(r,t,re,time_zero,shift,new_t=np.array([-500,-250,0,250,500,1000,1500,2000,500000]),s=None):    
    t_array=np.tile(t.reshape(t.size,1),(1,r.shape[1]))
    r_new=zeros((new_t.size,r.shape[1]))
    for i in range(r.shape[1]):
        #print i
        t_array[:,i]-=time_zero[i]-shift
        k=splrep(t_array[:,i],r[:,i],re[:,i],s=s)
        r_new[:,i]=splev(new_t,k)
    return r_new


def legend_format(l):
    return [str(i/1000.)+ ' ps' for i in l]

def make_legend(p,err,n):
    dig=np.floor(log10(err))
    l=[]
    for i in range(2,n+2):
        val=str(around(p[i],-int(dig[i])))
        erri=str(ceil(round(err[i]*10**(-dig[i]),3))*10**(dig[i]))
        l.append('$\\tau_'+str(int(i-1))+'='+val+'\\pm '+erri+' $ps')
    return l
    

def make_legend_noerr(p,err,n):
    dig=np.floor(log10(err))
    l=[]
    for i in range(2,n+2):
        val=str(around(p[i],-int(dig[i])))
        erri=str(ceil(round(err[i]*10**(-dig[i]),3))*10**(dig[i]))
        l.append('$\\tau_'+str(int(i-1))+'$='+val+' ps')
    return l

class model:
    def __init__(self,x):
        self.x=x
    def fit(self,data):        
        x=data
        return np.polyfit(x[:,0],x[:,1],3)
    
    def get_error(self,data,model):        
        x=data.T
        return (np.polyval(model,x[0,:])-x[1,:])**2



def savitzky_golay(y, window_size, order, deriv=0):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the swww.http.com//scipy-central.org/item/37/1/generating-confidence-intervals-via-model-comparsionsmoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t,files=glob.glob(name+'*dat?.npy') np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannerypsd1D = radialProfile.azimuthalAverage(psd2D)
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m, y, mode='valid')

def calc_error(args):
    p, cov, info, mesg, success = args
    chisq = sum(info["fvec"] * info["fvec"])
    dof = len(info["fvec"]) - len(p)
    sigma = np.array([np.sqrt(cov[i, i]) * np.sqrt(chisq / dof) for i in range(len(p))])
    return p, sigma


#a=ransac.ransac(x,model(x),40,15000,0.1,40)
#plot(polyval(a,arange(400)))
#plot(tn)
#tn1=polyval(a,arange(400))


#if __name__== '__main__':
#    import matplotlib.pyplot as plt
#    k=[250, 399, 0]
#    (t,w,s, se,r, re)=testload()
#    r=calc_od(r)
#    a=subtract_background(r)
#    rf=re*a
#    plt.plot(t,  a[:, k], t, rf[:, k]+a[:, k], t,-rf[:, k]+a[:, k] )
#    plt.show()
#    
#        
def w_rgb(w):
    if w >= 380 and w < 440:
        R = -(w - 440.) / (440. - 350.)
        G = 0.0
        B = 1.0
    elif w >= 440 and w < 490:
        R = 0.0
        G = (w - 440.) / (490. - 440.)
        B = 1.0
    elif w >= 490 and w < 510:
        R = 0.0
        G = 1.0
        B = -(w - 510.) / (510. - 490.)
    elif w >= 510 and w < 580:
        R = (w - 510.) / (580. - 510.)
        G = 1.0
        B = 0.0
    elif w >= 580 and w < 645:
        R = 1.0
        G = -(w - 645.) / (645. - 580.)
        B = 0.0
    elif w >= 645 and w <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    if (w>700.):
        s=.3+.7* (780.-w)/(780.-700.)
    elif (w<420.):
        s=.3+.7*(w-380.)/(420.-380.)
    else:
        s=1.
    R,G,B=(np.array((R,G,B))*s)**0.8
    return R,G,B
    
def equal_color(plots1,plots2):    
    if len(plots1)!=len(plots2):
        raise Not_same_length
    for (plot1,plot2) in zip(plots1,plots2):
        plot2.set_color(plot1.get_color())
