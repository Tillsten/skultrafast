import numpy as np
from scipy.optimize import leastsq
from scipy.stats import f
import matplotlib.pyplot as plt

def fix_para(fit_func,idx,val):
    """
    Given a function, an index and a value it returns a
    function without the parameter and idx and fixes the
    missing parameter the val.
    So it is a functools.partial for list-takinge functions.
    """
    def fixed_func(a):
        try:
            a=list(a)
        except:
            a=[a]
        a.insert(idx,val)
        return fit_func(a)
    return fixed_func

def sumsq(a):
    """Squares and sums an array"""
    return (a**2).sum()

def f_compare(fit_func,para,best_chi,num_points):
    """Returns th[3,5.,-5,15.]e probalitiy for two given parameter sets"""
    P=len(para)
    N=(num_points-P)
    return f.cdf((N-P)/P*(sumsq(fit_func(para))-best_chi)/best_chi,P,N-P)


def chi_search(fit_func,best,step_frac=1e-2,max_iter=1000,verbose=False):
    """
    For given function and found optimum the function calculates
    confindance intervals for every parameter via model
    comparsion.
    """

    best_res=fit_func(best)
    S0=sumsq(best_res)
    local_f_compare=lambda para: f_compare(fit_func, para, S0, best_res.size)
    list_of_arrays=[]
    for i in range(len(best)):

        hist=[]
        x0=list(best)
        val=x0.pop(i)
        org_val=val.copy()
        step=val*step_frac
        ret=0
        hist.append([val,ret]+list(best))

        def do_step(x0):
            fixed_func=fix_para(fit_func,i,val)
            new_para=list(leastsq(fixed_func,x0)[0])
            x0_new=new_para[:]
            new_para.insert(i,val)
            ret=local_f_compare(new_para)
            hist.append([val,ret]+new_para)
            return ret, x0_new

        iteration=0
        while ret<0.99 and iteration<max_iter:
            val+=step
            ret,x0=do_step(x0)
            iteration=iteration+1
            if verbose:
                print i, val,ret

        ret=0
        val=org_val
        x0=list(best)
        val=x0.pop(i)

        iteration=0
        while ret<0.99 and iteration<max_iter:
            val-=step
            ret,x0=do_step(x0)
            iteration=iteration+1
            if verbose:
                print i, val,ret

        hist_array=np.array(hist)
        idx=np.argsort(hist_array[:,0],0)
        hist_array=hist_array[idx,:]
        list_of_arrays.append(hist_array)
    return list_of_arrays

def plot2c(a,true_val):
    """Helper function for plottion"""
    plt.plot(a[:,0],a[:,1])
    plt.axhline(0.95)
    plt.text(0.5,0.89,r'$2\sigma$',transform=plt.gca().transAxes)
    plt.axhline(0.68)
    plt.text(0.5,0.62,r'$1\sigma$',transform=plt.gca().transAxes)
    plt.axvline(true_val,ymax=0.3,color='k')
    #plt.text(true_val, 0.4, ')

def calc_error(args):
    """Function to calculate the errors from the estimated covariance matrix.

    Takes the output from leastsq with full_output=1 as argument.
    """
    p, cov, info, mesg, success = args
    chisq = sum(info["fvec"] * info["fvec"])
    dof = len(info["fvec"]) - len(p)
    sigma = np.array([np.sqrt(cov[i, i]) * np.sqrt(chisq / dof) for i in range(len(p))])
    return p, sigma

if __name__=='__main__':
    #Generate test data
    np.random.seed(0)
    x=np.linspace(5,10,10)
    #Our data a*x+b with a=3, b=3
    y=3*x+3
    yn=y+np.random.randn(y.size)*2
    #The residual function to fit.
    def linear_fit(p):
        return (p[0]*x+p[1])-yn

    #Fit it
    out=leastsq(linear_fit,(2,2),full_output=1)
    #Calc error from cov. matrix
    p,s=calc_error(out)
    #Plot
    plt.subplot(221)
    plt.plot(x,y,'--')
    plt.plot(x,yn,'ko')
    plt.plot(x,linear_fit(out[0])+yn)
    plt.xlim(0,10)
    plt.xlabel('x')
    plt.ylabel('y')

    #Using our method
    b=chi_search(linear_fit,out[0])
    #Display results
    plt.subplot(222)
    plot2c(b[0],3)
    plt.hlines(0.68,p[0]-s[0],p[0]+s[0],lw=2)
    plt.xlabel('a')
    plt.ylabel('probabilty')
    plt.subplot(223)
    plot2c(b[1],3)
    plt.hlines(0.68,p[1]-s[1],p[1]+s[1],lw=2)
    plt.xlabel('b')
    plt.ylabel('probabilty')


    #Show prob(a,b) surface:
    plt.subplot(224)
    S0=sumsq(linear_fit(out[0]))
    pos_a=np.linspace(b[0][:,0].min(),b[0][:,0].max(),40)
    pos_b=np.linspace(b[1][:,0].min(),b[1][:,0].max(),40)
    X,Y=np.meshgrid(pos_a,pos_b)
    paras=np.dstack((X,Y))
    ap=lambda p: f_compare(linear_fit,p,S0,y.size)
    chisq=np.apply_along_axis(ap,2,paras)
    plt.hot()
    plt.contourf(pos_a,pos_b,chisq,30)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.tight_layout()
    plt.show()
