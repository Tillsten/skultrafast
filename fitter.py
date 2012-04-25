# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from scipy.special import erfc #errorfunction
from scipy.optimize import leastsq,brentq,bisect#, fmin_bfgs  #lm function
from scipy.stats import f #fisher
from scipy.linalg import qr, lstsq
import scipy.sparse.linalg as li
#from scipy import optimize as opt
#import openopt as oo
import dv
#import cma
#a=ne.NumExpr("exp(k*(w*w*k/(4.0)-t))/(w*np.sqrt(2*np.pi))*0.5*erfc(-t/w+w*k/(2.0))")
sq2=np.sqrt(2)


def _fold_exp(tt,w,tz,tau):
    """
    Returns the values of the folded exponentials for given parameters.

    Parameters
    ----------
    tt:  ndarray(N)
        Array containing the time-coordinates
    w:  float
        The assumed width/sq2    
    tz: float
        The assumed time zero.    
    tau: ndarray(M)
        The M-decay rates.
        
    Returns
    -------
    y: ndarray
       Folded Exponentials for given tau's.
        
"""

    k=1/(tau[:,None])
    t=tt+tz
    y=np.exp(k*(w*w*k/(4.0)-t))/(w*np.sqrt(2*np.pi))*0.5*erfc(-t/w+w*k/(2.0))
    return y

def _exp(tt,w,tz,tau):
    """
    Returns the values of the exponentials for given parameters.

    Parameters
    ----------
    tt:  ndarray(N)
        Array containing the time-coordinates
    w:  float
        The assumed width/sq2    
    tz: float
        The assumed time zero.    
    tau: ndarray(M)
        The M-decay rates.
        
    Returns
    -------
    y: ndarray
       Exponentials for given tau's.
        
    """
    t=tt+tz
    y=np.exp(-t/(tau[:,None]))
    return y


def _coh_gaussian(t,w,tz):
    """Models artefacts proptional to a gaussian and it's derivates
    
    Parameters
    ----------
    t:  ndarray
        Array containing the time-coordinates
    w:  float
        The assumed width/sq2    
    tz: float
        The assumed time zero.

    Returns
    -------
    y:  ndarray (len(t), 4)
        Array containing a gaussan and it the scaled derivates, 
        each in its own coulumn.
        
    """
    w=w*sq2
    tt=t+tz
    y=np.tile(np.exp(-0.5*(tt/w)**2)/(w*np.sqrt(2*np.pi)),(4,1)).T
    y[:,1]*=(-tt/w**2)
    y[:,2]*=(tt**2/w**4-1/w**2)
    y[:,3]*=(-tt**3/w**6+3*tt/w**4)
    #y/=np.max(np.abs(y),0)
    return y

class Fitter(object):
    """ The fit object, takes all the need data and allows to fit it.
    
    Parameters
    ----------
    wl : ndarray(M)
        Array containing the wavelenthscorridnates
    t :  ndarray(N)
        Array containing the time-coordinates
    data :  ndarry(N,M)
        The data to fit.
    num_exponentials : int
        Number of exponentials to fit.
    model_coh : bool
        If to model coherent artefacts at the time zero, defaults to False
    bounds : float
        Bounds to use for constraint fitting of the linear coeffecients, is
        only used in con_model. 
    
    
    """
    def __init__(self,wl,t,data,num_exponentials,model_coh=False,bound=1000.):        
        self.t=t
        self.wl=wl
        self.model_coh=model_coh
        if model_coh:
            self.x_vec=np.zeros((self.t.size,num_exponentials+4))
        else:
            self.x_vec=np.zeros((self.t.size,num_exponentials))
        self.data=data
        self.num_exponentials=num_exponentials
        #self.one=np.identity(t.size)
        self.last_spec=None
        self.bound=bound
        self.weights=1.

    def model(self,para,fixed=None):
        """ Returns the fit for given psystemeqarameters.

        para has the following form:
        [xc,w,tau_1,...tau_n]"""
        self.build_xvec(para,fixed)
        self.c=lstsq(self.x_vec,self.data)[0]
        self.m=np.dot(self.c.T,self.x_vec.T)
        
#    def modelf(self,para,fixed=None):
#        """ Returns the fit for given psystemeqarameters.
#
#        para has the following form:
#        [xc,w,tau_1,...tau_n]"""
#
#        self.build_xvec(para,fixed)
#        xl=self.x_vec.shape[1]
#        r=np.zeros((xl,self.data.shape[1]))
#        for i in range(self.data.shape[1]):
#            a=li.lsqr(self.x_vec,self.data[:,i],0.001)
#            r[:,i]=a[0]
#        self.c=r
#        self.m=self.x_vec.dot(r).T


    def con_model(self,para,fixed=None):
        self.last_para=para
        #print para
        self.build_xvec(para,fixed)
        xl=self.x_vec.shape[1]
        lb=-self.bound*np.ones(xl)

        ub=-lb
        #self.last_spec=opt.fmin_slsqp(self.fcon,x0,bounds=bounds)
        r=np.zeros((xl,self.data.shape[1]))
        for i in range(self.data.shape[1]):
            a=oo.LLSP(self.x_vec,self.data[:,i],ub=ub,lb=lb)
            a.iprint=-1
            a.solve('bvls')
            r[:,i]=a.xf
        self.last_spec=r
        self.last_res=self.x_vec.dot(r)-self.data
        #self.last_spec=opt.fmin_l_bfgs_b(self.fcon,x0,approx_grad=1,bounds=bounds)[0]
        #self.last_spec=opt.fmin_cobyla(self.fcon,x0,lambda x: min(x-40.))
        return (self.last_res).flatten()

    def build_xvec(self,para,fixed=None):
        """Build the base (the folded functions) for given parameters.
        
        Parameters
        ----------
        """

        if fixed:
            para=list(para)
            for (i,j) in fixed:
                para.insert(i,j)
        self.last_para=para
        para=np.array(para)
       
        if self.model_coh:
            self.x_vec[:,-4:]=_coh_gaussian(self.t,para[1],para[0])

        self.x_vec[:,:self.num_exponentials]=_fold_exp(self.t,para[1],para[0],(para[2:])).T
        self.x_vec=np.nan_to_num(self.x_vec)

    def res(self,para,fixed=None):
        """Return the residuals for given parameters."""
        self.model(para,fixed)
        return ((self.data-self.m.T)/self.weights).flatten()

    def res_sum(self,para,fixed=None):
        """Returns the squared sum of the residuals for given parameters"""
        return np.sum(self.res(para,fixed)**2)

    def ret_m(self,para):
        """Return the flattend fit for given parameter"""
        self.model(para)
        return self.m.flatten()


    def varpro(self, para,fixed=None):
        """Variable Projection functional"""
        self.build_xvec(para,fixed)
        #q=qr(self.x_vec)[0]
        #q1,q2=q[:,:self.x_vec.shape[1]],q[:,self.x_vec.shape[1]:]
        #qm=np.dot(q2,q2.T)
        res=qr(self.x_vec)[0][:,self.x_vec.shape[1]:].T.dot(self.data)
        return 0.5*res.flatten()

    def res_sum_logscaler(self,para,fixed=None):
        para[1:]=np.exp(para[1:])
        o=self.res_sum(para,fixed)
        if False:#np.any(np.abs(self.c)>500):
            return np.nan
        else:
            return o

    def start_fit(self, x0, fixed=None,**kwargs):
        """
        Starts the fit for given x0 and fixed parameters.
        Returns found x and it's errors.
        """
        if fixed!=None:
            print 'isFIxed', fixed
            best=leastsq(self.res,x0,fixed, full_output=True,**kwargs)
        else:
            best=leastsq(self.res,x0, full_output=True,**kwargs)
        print 'Chi^2:  ', (self.res(best[0],fixed)**2).sum()
        try:
            p,c=dv.calc_error(best)        
            for (pi,ci) in zip(p,c):
                print "{0: .3f} +- {1:.4f}".format(pi,ci)
            return p,c,best
        except TypeError:
            for pi in best[0]:
                print "{0: .3f}".format(pi)
            return best[0]                
                

    def start_cmafit(self,x0,fixed=None,restarts=2):
        import cma
        
        out=cma.fmin(self.res_sum,x0,2,verb_log=0,verb_disp=50,
                     restarts=restarts,tolfun=1e-6,tolfacupx=1e9,
                     args=fixed)
        for pi in (out[0]):
            print "{0: .3f} +- {1:.4f}".format(pi,np.exp(pi))

        return out

    def chi_search(self,best, fixed=None, step_frac=1e-3, ignore_last=False):        
        S0=self.res_sum(best)
        def f_compare(para):
            """Returns the F-Value for two given parameter sets"""            
            P=self.num_exponentials
            N=(self.t.size-P)*(self.data.shape[1])
            return f.cdf((N-P)/P*(self.res_sum(para)-S0)/S0,P,N-P)
        if fixed==None: fixed=[]
        a=best.copy()
        
        l_arr=[]
        if ignore_last:
            tmp=1
        else: 
            tmp=0
            
        for i in range(len(a)-tmp):             
            l=[]                    
            a=list(best)
            val=a.pop(i)
            oval=val.copy()
            step=val*step_frac
            #step=max(step,0.05)
            ret=0
            sigmas=[0.997,0.95,0.68]
            def prob_func(val,prob,a):                
                new_p=list(leastsq(self.res,a,[(i,val)]+fixed)[0])                
                a[:]=new_p[:]
                new_p.insert(i,val)     
                ret=f_compare(new_p)
                #               
                return prob-ret
                
            def gen_ubound(val):
                while prob_func(val,0.999,a)>0:
                    val=val*1.1
                return val
            def gen_lbound(val):
                while prob_func(val,0.999,a)>0:
                    val=val*0.9
                return val
            uoval=gen_ubound(oval)
            loval=gen_lbound(oval)
            
            for prob in sigmas:
                fs=lambda x: prob_func(x,prob,a)
                try:
                    
                    uoval,r=brentq(fs,oval,uoval,
                             disp=True,rtol=0.0001,
                             full_output=True)
                    
                    loval,r2=brentq(fs,loval,oval,
                             disp=True,rtol=0.0001,
                             full_output=True)
                    l.append([prob,loval,uoval])
                    #print prob, 'res',fs(uoval),fs(oval)                    
                    print l[-1]
                except:
                    #print prob, 'schief',fs(uoval),fs(oval)
                    print "ging was schief"
                
            l_arr.append(l)
            #subplot(4,2,i+1)
            #l.append((val,ret))    
            
#            while ret<0.99:        
#                val+=step        
#                new_p=list(leastsq(self.res,a,[(i,val)])[0])
#                a=new_p[:]
#                new_p.insert(i,val)      
#                
#                ret=f_compare(new_p)
#                l.append((val,ret))
#                print i, val,ret
#                print np.round(new_p,2)
#            
#            a=list(best)
#            val=a.pop(i)
#            ret=0
#            val=oval
#            while ret<0.99:        
#                val-=step        
#                new_p=list(leastsq(self.res,a,[(i,val)])[0])
#                a=new_p[:]
#                new_p.insert(i,val)        
#                ret=f_compare(new_p)
#                l.append((val,ret))
#                print i, val,ret
#                print np.round(new_p,2)
#
#                
#            tmp=np.array(l)     
#            arg=np.argsort(tmp[:,0],0)
#            tmp=tmp[arg,:]
#            l_arr.append(tmp)
        return l_arr
        
    
    def start_pymc(self,x0,bounds):
        import pymc            
        rs=[(pymc.Uniform('r'+str(i),lower,upper)) for (i,(lower,upper)) in enumerate(bounds)]            
        z0=pymc.Uniform('z0',-1,1)
        sig=pymc.Uniform('sig',0,0.15)
        tau=pymc.Uniform('tau',0,25)
        tau.value=20
                 
        H=lambda x: self.model(x)
        @pymc.deterministic
        def mod(z0=z0,sig=sig,rs=rs):
            x=np.array([z0,sig]+rs)       
            H(x)
            return self.m.T
        
        res=pymc.Normal('res',mu=mod, observed=True, tau=tau, value=self.data)
        mo=pymc.Model(set([z0,sig,res,tau]+rs))
      
        return mo


unitdict={'x': ' nm', 'y': ' ps', 'z': '$\\Delta$oD'}
import matplotlib.pyplot as plt

def plot_das(fitter,plot_fastest=False,plot_coh=False,normed=False):
    fitter.last_para[2:]=np.sort(fitter.last_para[2:])
    fitter.res(fitter.last_para)
    if plot_coh and fitter.model_coh:
        ulim=fitter.num_exponentials
    else:
        ulim=-4
    
    if plot_fastest:
        llim=0
    else:
        llim=1
    print llim, ulim
    dat_to_plot= fitter.c.T[:,llim:ulim]
    if normed: dat_to_plot=dat_to_plot/np.abs(dat_to_plot).max(0)
    plt.plot(fitter.wl,dat_to_plot,lw=2)
    plt.autoscale(1,tight=1)
    plt.legend(np.round(fitter.last_para[2+llim:],2))
    plt.xlabel(unitdict['x'])
    plt.ylabel(unitdict['z'])

def plot_diagnostic(fitter):   
    residuals=fitter.data-fitter.m.T
    u,s,v=np.linalg.svd(residuals)
    plt.subplot2grid((3,3),(0,0),2,3).imshow(residuals,aspect='auto')
    plt.subplot2grid((3,3),(2,0)).plot(u[:,0])
    plt.subplot2grid((3,3),(2,1)).plot(v.T[:,0])
    ax=plt.subplot2grid((3,3),(2,2))
    ax.stem(range(1,11),s[:10])
    ax.set_xlim(0,12)
    
def plot_spectra(fitter,tp=None,num_spec=8):
    t=fitter.t
    tmin,tmax=t.min(),t.max()
    if tp==None: tp=np.logspace(np.log10(0.100),np.log10(tmax),num=num_spec)
    tp=np.round(tp,2)
    specs=dv.interpol(fitter.data,t,np.zeros(fitter.data.shape[1]),0,tp)
    plt.plot(fitter.wl,specs.T)
    plt.legend(tp,ncol=2)
    plt.autoscale(1,tight=1)
    plt.xlabel(unitdict['x'])
    plt.ylabel(unitdict['z'])

    
def plot_transients(fitter,wls, plot_fit=True,scale='linear'):
    wls=np.array(wls)
    idx=np.argmin(np.abs(wls[:,None]-fitter.wl[None,:]),1)
    plt.plot(fitter.t, fitter.data[:,idx],'^')
    plt.legend([unicode(i)+u' '+unitdict['x'] for i in np.round(fitter.wl[idx])])
    if plot_fit and hasattr(fitter,'m'):
        plt.plot(fitter.t,fitter.m.T[:,idx],'k')
    plt.autoscale(1,tight=1)
    plt.xlabel(unitdict['y'])
    plt.ylabel(unitdict['z'])
    if scale!='linear':
        plt.xscale(scale)
        
    
if __name__=='__main__':
    t=np.linspace(-1,30,500)
    coef=np.zeros((2,400))
    coef[0,:]=-np.arange(400)
    coef[1,:]=np.arange(-200,200)**2/100.
    
    g=Fitter(np.arange(400),t, 0,2,False)
    g.build_xvec([0.1,0.3,5,16])
    dat=np.dot(g.x_vec,coef)
    
    dat+=10*(np.random.random(dat.shape)-0.5)
    dat=dat*(1+(np.random.random(dat.shape)-0.5)*0.20)
    g=Fitter(np.arange(400),t,dat,2,False)
    x0=[0.5,0.2,4,20]
    #a=g.start_pymc(x0)
    #a=g.start_cmafit(x0)
    a=g.start_fit(x0)
    #ar=g.chi_search(a[0])
    import matplotlib.pyplot as plt
    #
##    def plotxy(a):
##        plt.plot(a[:,0],a[:,1])
##    #
##    for i in range(len(a[0])-1):
##        plt.subplot(2,2,i+1)
##        plotxy(ar[i])
    #plt.tight_layout()
    #plot_das(g,1)
    plot_diagnostic(g)
    #plot_spectra(g)
    #wls=[30,70,100]
    #plot_transients(g,wls)
    plt.show()
    #best=leastsq(g.varpro,x0, full_output=True)
