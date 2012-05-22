# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from scipy.special import erfc #errorfunction
from scipy.optimize import leastsq,brentq#, fmin_bfgs  #lm function
from scipy.stats import f #fisher
from scipy.linalg import qr, lstsq
import scipy.sparse.linalg as li
#from scipy import optimize as opt
#import openopt as oo
import dv
import lmfit
sq2=np.sqrt(2)
BASE_WL=550.

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
    ws=w
    k=1/(tau[:,None])
    t=tt+tz    
    y=np.exp(k*(ws*ws*k/(4.0)-t))/(ws*np.sqrt(2*np.pi))*0.5*erfc(-t/ws+ws*k/(2.0))
    y/=np.max(np.abs(y),0)
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
    w=w/sq2
    tt=t+tz
    y=np.tile(np.exp(-0.5*(tt/w)**2)/(w*np.sqrt(2*np.pi)),(4,1)).T
    y[:,1]*=(-tt/w**2)
    y[:,2]*=(tt**2/w**4-1/w**2)
    y[:,3]*=(-tt**3/w**6+3*tt/w**4)
    y/=np.max(np.abs(y),0)
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
    model_coh : bool
        If to model coherent artefacts at the time zero, defaults to False
    bounds : float
        Bounds to use for constraint fitting of the linear coeffecients, is
        only used in con_model.


    """
    def __init__(self,wl,t,data,
                 model_coh=False,model_disp=0,bound=1000.):
        self.t=t
        self.wl=wl
        self.model_coh=model_coh
        self.model_disp=model_disp
        self.data=data
        
        #self.one=np.identity(t.size)
        self.last_spec=None
        self.bound=bound
        self.weights=1.
        if model_disp:
            self.org=data[:]
            self.minwl=np.min(wl)
            self.used_disp=np.zeros(model_disp)
        
    def model(self,para,fixed=None):
        """ Returns the fit for given psystemeqarameters.

        para has the following form:
        [xc,w,tau_1,...tau_n]"""        
        self.last_para=para
    
        if self.model_disp:
            if  np.any(para[:self.model_disp]!=self.used_disp):
                self.tn=np.poly1d(para[:self.model_disp]+[0])(self.wl-self.minwl)
                self.data=dv.interpol(self.org,self.t,self.tn,1., self.t)
                self.used_disp[:]=para[:self.model_disp]

            para=para[self.model_disp:]
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


    def constrained_model(self,para,fixed=None):
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

        """
        if fixed:
            para=list(para)
            for (i,j) in fixed:
                para.insert(i,j)

        para=np.array(para)
        self.num_exponentials=para.size-2
        if self.model_coh:
            self.x_vec=np.zeros((self.t.size,self.num_exponentials+4))
            self.x_vec[:,-4:]=_coh_gaussian(self.t,para[1],para[0])
            self.x_vec[:,:-4]=_fold_exp(self.t,para[1],para[0],(para[2:])).T
        else:
            self.x_vec=_fold_exp(self.t,para[1],para[0],(para[2:])).T
                        
           

        
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


    def start_lmfit(self,x0,fixed_names=[],lower_bound=0.3):        
        p=lmfit.Parameters()
        for i in range(self.model_disp):
            p.add('p'+str(i),x0[i])                       
        x0=x0[self.model_disp:]
        p.add('x0',x0[0])
        p.add('w',x0[1],min=0)
        for i,tau in enumerate(x0[2:]):
            p.add('t'+str(i),tau,vary=True,min=lower_bound)
        
        for k in fixed_names:
            p[k].vary=False
        
        def res(p):
            x=[k.value for k in p.values()]            
            return self.res(x)
            
        return lmfit.Minimizer(res,p)
            

    def start_cmafit(self,x0,fixed=None,restarts=2):
        import cma

        out=cma.fmin(self.res_sum,x0,2,verb_log=0,verb_disp=50,
                     restarts=restarts,tolfun=1e-6,tolfacupx=1e9,
                     args=fixed)
        for pi in (out[0]):
            print "{0: .3f} +- {1:.4f}".format(pi,np.exp(pi))

        return out



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


def f_compare(Ndata, Nparas, new_chi, best_chi, Nfix=1.):
    """
    Returns the probalitiy for two given parameter sets.
    Nfix is the number of fixed parameters.
    """

    # print new_chi, best_chi, Ndata, Nparas

    Nparas = Nparas + Nfix
    return f.cdf((new_chi / best_chi - 1) * (Ndata - Nparas) / Nfix,
                 Nfix, Ndata - Nparas)


def my_f(N, P, chi, old_chi, Nfix=1., lin_dof=800):
    print "klappt"

    return f_compare(N, P+lin_dof, chi, old_chi, Nfix)


if __name__=='__main__':

    coef=np.zeros((2,400))
    coef[0,:]=-np.arange(-300,100)**2/100.
    coef[1,:]=np.arange(-200,200)**2/100.
    t=np.linspace(0,30,300)
    g=Fitter(np.arange(400),t, 0,False)
    g.build_xvec([0.1,0.3,5,16])
    dat=np.dot(g.x_vec,coef)

    dat+=10*(np.random.random(dat.shape)-0.5)
    dat=dat*(1+(np.random.random(dat.shape)-0.5)*0.20)
    g=Fitter(np.arange(400),t,dat,2,False,False)
    x0=[0.5,0.2,4,20]
#    #a=g.start_pymc(x0)
#    #a=g.start_cmafit(x0)
    a=g.start_lmfit(x0)
    a.leastsq()
    lmfit.printfuncs.report_errors(a.params)
#    #ar=g.chi_search(a[0])
#    import matplotlib.pyplot as plt
    #
##    def plotxy(a):
##        plt.plot(a[:,0],a[:,1])
##    #
##    for i in range(len(a[0])-1):
##        plt.subplot(2,2,i+1)
##        plotxy(ar[i])
    #plt.tight_layout()
    #plot_das(g,1)
    #plot_diagnostic(g)
    #plot_spectra(g)
    #wls=[30,70,100]
    #plot_transients(g,wls)
    #plt.show()
    #best=leastsq(g.varpro,x0, full_output=True)
    