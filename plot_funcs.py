# -*- coding: utf-8 *-*
units = {'x': ' nm', 'y': ' ps', 'z': r'$\Delta$OD'}
title = ""
import matplotlib.pyplot as plt
import numpy as np
import dv, data_io, zero_finding

plt.rcParams['font.size']=9
plt.rcParams['legend.fontsize'] = 'small'
#plt.rcParams['legend.borderpad'] = 0.1
#plt.rcParams['legend.columnspacing'] = 0.3
#plt.rcParams['legend.labelspacing'] = 0.3
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['image.cmap'] = 'PRGn'

def plot_das(fitter, plot_fastest=0, plot_coh=False,
             normed=False, sas=False, const=False):
    """Plots the decay-asscoiated  """        
    num_exp = fitter.num_exponentials
    #fitter.last_para[-num_exp:] = np.sort(fitter.last_para[-num_exp:])
    
    
    if plot_coh or not fitter.model_coh:
        ulim = fitter.c.size
    else:
        ulim = fitter.num_exponentials

    llim = plot_fastest
    
    dat_to_plot= fitter.c[:, llim:ulim]
    if sas:
        dat_to_plot = -dat_to_plot.sum(1)[:,None]+np.cumsum(dat_to_plot, 1)
    if normed: 
        dat_to_plot = dat_to_plot/np.abs(dat_to_plot).max(0)
    plt.plot(fitter.wl, dat_to_plot, lw=2)
    plt.autoscale(1, tight=1)
    plt.axhline(0, color='grey', zorder=-10, ls='--')
    leg = np.round(fitter.last_para[1 + llim + fitter.model_disp:], 1)
    leg = [str(i)+ units['y'] for i in leg]
    if const:
        leg[-1] = 'const'
        
    plt.legend(leg, labelspacing=0.25)
    
    plt.xlabel(units['x'])
    plt.ylabel(units['z'])
    if title:
        plt.title(title)
        
def plot_pol_das(fitter, plot_fastest=0, plot_coh=False, normed=False):    
    """Plots the decay-asscoiated spectra for polarized data"""        
    t_slice = fitter.model_disp+2
    fitter.last_para[t_slice:] = np.sort(fitter.last_para[t_slice:])
    fitter.res(fitter.last_para)
    
    if plot_coh and fitter.model_coh:
        ulim = fitter.num_exponentials
    else:
        ulim =- 4

    llim = plot_fastest
    
    dat_to_plot= fitter.c[:,llim:ulim]
    if normed: 
        dat_to_plot = dat_to_plot / np.abs(dat_to_plot).max(0)
    half_idx = dat_to_plot.shape[0]/2       
    p1 = plt.plot(fitter.wl, dat_to_plot[:half_idx, :], lw=2)
    p2 = plt.plot(fitter.wl, dat_to_plot[half_idx:, :], '--', lw=2)
    dv.equal_color(p1, p2)
    plt.autoscale(1, tight=1)
    plt.axhline(0, color='grey', zorder=-10, ls='--')
    leg = np.round(fitter.last_para[2 + llim + fitter.model_disp:], 2)
    plt.legend([str(i)+ units['y'] for i in leg], labelspacing=0.25)
    plt.xlabel(units['x'])
    plt.ylabel(units['z'])
    if title:
        plt.title(title)
        

def plot_diagnostic(fitter):
    residuals = fitter.residuals
    u, s, v = np.linalg.svd(residuals)
    normed_res = residuals / np.std(residuals, 0)
    plt.subplot2grid((3, 3), (0, 0), 2, 3).imshow(normed_res, vmin=-0.5,
                                             vmax=0.5, aspect='auto')
    plt.subplot2grid((3, 3), (2, 0)).plot(fitter.t, u[:,:2])
    plt.subplot2grid((3, 3), (2, 1)).plot(fitter.wl, v.T[:,:2])
    ax=plt.subplot2grid((3, 3), (2, 2))
    ax.stem(range(1, 11), s[:10])
    ax.set_xlim(0, 12)

def plot_spectra(fitter, tp=None, pol=False, num_spec=8, use_m=False,
                 cm='Spectral', lw=1.5, tmin=None, tmax=None):
    """
    Plots the transient spectra of an fitter object.
    """
    t = fitter.t
    if tmin is None:
        tmin = t.min() 
    if tmax is None:
        tmax = t.max() 
    if tp is None:         
        tp = np.logspace(np.log10(max(0.100, tmin)), np.log10(tmax), num=num_spec)
        #tp = np.hstack([-0.5, -.1, 0, tp])
    tp = np.round(tp, 2)    
    t0 = fitter.last_para[fitter.model_disp]
    
    if use_m:
        data_used = fitter.m.T
    else:
        data_used = fitter.data
    
    if hasattr(fitter, 'tn'):
        tn = fitter.tn
        t0 = 0.
    else:
        tn = np.zeros(fitter.data.shape[1])

    specs = zero_finding.interpol(dv.tup(fitter.wl, fitter.t, data_used),
                             tn, t0, tp).data    
    
    p1 = plt.plot(fitter.wl, specs[:, :fitter.wl.size].T, lw=2*lw)    

    if cm:
        use_cmap(p1, cmap=cm) 
        
    
    if pol:
        p2 = plt.plot(fitter.wl, specs[:, fitter.wl.size:].T, lw=lw)    
        dv.equal_color(p1, p2)
    plt.legend([unicode(i)+u' '+units['y'] for i in np.round(tp,2)],
                ncol=1,  labelspacing=0.25)
    plt.axhline(0, color='grey', zorder=-10, ls='--')
    plt.autoscale(1, tight=1)
    plt.xlabel(units['x'])
    plt.ylabel(units['z'])
    if title:
        plt.title(title)


def plot_transients(fitter, wls, pol=False, plot_fit=True, scale='linear',
                    plot_res=False, ncol=2):
    wls = np.array(wls)
    idx = np.argmin(np.abs(wls[:,None]-fitter.wl[None,:]), 1)    
    names = [str(i) + u' ' + units['x'] for i in np.round(fitter.wl[idx])]
    
    if hasattr(fitter, 't_mat'):
        t = fitter.t_mat[:, idx]        
    else:
        t = fitter.t + fitter.last_para[0]
    
    data_to_plot =  fitter.data[:, idx]
    if plot_res: 
        data_to_plot -= fitter.model[:, idx]
    p1 = plt.plot(t, data_to_plot , '^')
    
    if pol: 
        p2 = plt.plot(t, fitter.data[:, idx + fitter.data.shape[1] / 2], 'o') 
        dv.equal_color(p1, p2)
    
    plt.legend(names, scatterpoints=1, numpoints=1, ncol=ncol)
    
    if plot_fit and hasattr(fitter,'model'):       
        plt.plot(t, fitter.model[:, idx], 'k')
        if pol:
            plt.plot(t, 
                     fitter.model[:, idx + fitter.data.shape[1] / 2], 'k')
            
    plt.autoscale(1, tight=1)
    plt.xlim(max(-0.3, t.min()))
    plt.xlabel(units['y'])
    plt.ylabel(units['z'])
    if scale != 'linear':
        plt.xscale(scale)
    if title:
        plt.title(title)

def plot_residuals(fitter, wls, scale='linear'):
    wls = np.array(wls)
    idx = np.argmin(np.abs(wls[:, None] - fitter.wl[None, :]), 1)
    plt.plot(fitter.t, fitter.residuals[:, idx], '-^')
    plt.legend([unicode(i) + u' ' + units['x'] for i in np.round(fitter.wl[idx])],
                 labelspacing=0.25)
    plt.autoscale(1, tight=1)
    plt.xlabel(units['y'])
    plt.ylabel(units['z'])
    if scale != 'linear':
        plt.xscale(scale)
    if title:
        plt.title(title)
        
def a4_overview(fitter, fname, plot_fastest=1, linthresh=None, title=None):
    
    plt.ioff()
    tup_cor = zero_finding.interpol(fitter, fitter.tn, 0.0)
    
    f=plt.figure(1, figsize=(8.3, 12))
    plt.subplot(321)
    import matplotlib.colors as c
    if not linthresh:
        linthresh = abs(tup_cor.data).max() / 2.         
    m = max(abs(tup_cor.data.min()), abs(tup_cor.data.max()))
    sn = c.SymLogNorm(linthresh, vmin=-m, vmax=m)
    plt.pcolormesh(tup_cor.wl, tup_cor.t, tup_cor.data, norm=sn)    
    plt.yscale('symlog')
    plt.colorbar()
    plt.autoscale(1, tight=1)
    plt.ylim(max(-.3, fitter.t.min()))
    plt.subplot(322)
    plt.imshow(fitter.residuals / fitter.residuals.std(0), aspect='auto')
    plt.clim(-3,3)
    if title:    
        plt.title(title)
    plt.autoscale(1, tight=1)
    plt.subplot(323)
    plot_das(fitter, plot_fastest)
    plt.subplot(324)
    plot_das(fitter, plot_fastest, normed=True)
    plt.subplot(325)
    plot_spectra(fitter)
    plt.subplot(326)    
    wl = fitter.wl
    ind = [int(round(i)) for i in np.linspace(wl.min(), wl.max(), 10)]    
    plot_transients(fitter, ind, scale='symlog')
    plt.gcf().set_size_inches((8.2, 12))
    plt.tight_layout()
    plt.draw()
    
    if fname is not None:
        f.savefig(fname, dpi=150)    
        plt.close('all')
    
def plot_ltm_page(f, fname=None):
    from skultrafast import lifetimemap
    from matplotlib.colors import SymLogNorm
    plt.ioff()
    #plt.autoscale(True, 'both', tight=True)
    coefs, fit, taus, tup = lifetimemap.make_ltm(f)
    plt.figure(figsize=(12, 8.3))
    ax = plt.subplot2grid((2,2), (0,0), colspan=2)
    plt.sca(ax)

    m = max(abs(coefs.min()), abs(coefs.max()))
    sn = SymLogNorm(linthresh=3., vmin=-m, vmax=m)
    plt.pcolormesh(tup.wl, taus, coefs.T, cmap='coolwarm',norm=sn)
    #plt.clim(-abs(coefs).max(), abs(coefs).max())
    plt.yscale('log')
    plt.autoscale(True, 'both', tight=True)
    plt.colorbar()
    #plt.clabel('Amp.')
    
    
    plt.ylabel(r'$\tau$')
    plt.xlabel(r'wl / nm')
    plt.subplot(223)
    res = fit - tup.data
    plt.pcolormesh(tup.wl, tup.t, res)
    plt.subplot(224)
    plt.plot(tup.t, tup.data[:, ::20])
    plt.plot(tup.t, fit[:, ::20])
    plt.autoscale(True, 'both', tight=True)
    plt.xscale('log')
    plt.autoscale(True, 'both', tight=True)
    plt.xlabel('t')
    plt.ylabel(units['y'])
    plt.tight_layout()
    #plt.show()
    if fname:
        plt.savefig(fname, dpi=300)
        plt.close()
    
#f = zero_finding.interpol(g, g.tn)
#c, f, taus, tup = make_ltm(f)    
#plot_ltm_page(g)    
def a4_overview_second_page(fitter, para, perp, fname, linthresh=None):
    import matplotlib.gridspec as gs
    plt.clf()
    plt.ioff()
    fig, axs = plt.subplots(3, 1, figsize=(8.3, 12))
    plt.sca(axs[0])
    plot_map(para, linthresh)
    plt.title('parallel')
    
    plt.sca(axs[1])
    plot_map(perp, linthresh)
    plt.title('perp')
    
    plt.sca(axs[2])
    
    plt.show()
def plot_map(tup_cor, linthresh):
    import matplotlib.colors as c
    if not linthresh:
        linthresh = abs(tup_cor.data).max() / 2.         
    m = max(abs(tup_cor.data.min()), abs(tup_cor.data.max()))
    sn = c.SymLogNorm(linthresh, vmin=-m, vmax=m)

    plt.pcolormesh(tup_cor.wl, tup_cor.t, tup_cor.data, norm=sn)
    plt.yscale('symlog')
    plt.colorbar()
    plt.autoscale(1, tight=1)
    plt.ylim(max(-.3, tup_cor.t.min()))
    
def _plot_zero_finding(tup, raw_tn, fit_tn, cor):
    ax1 = plt.subplot(121)
    ax1.plot(tup.wl, raw_tn)    
    ax1.plot(tup.wl, fit_tn)    
    ax1.pcolormesh(tup.wl, tup.t, tup.data)
    ax1.set_ylim(fit_tn.min(), fit_tn.max())
    ax2 = plt.subplot(122)
    ax2.pcolormesh(cor.wl, cor.t, cor.data)
    ax2.set_ylim(fit_tn.min(), fit_tn.max())
    
    
    

def sig_ratios(fitter, fname=None, tmax=300,
               tmin = 0.1,
               do_fit=True, start_taus=None):
    if not start_taus:
        start_taus = [0.5, 11]
        
    t, pos, neg, pn, total = dv.calc_ratios(fitter, tmin=tmin, tmax=tmax)    
    labels = ['Positive / Negative', 'Positive',
              'Negative', 'Total']
    i = 0

    for l, y in zip(labels, (pn, pos, neg, total)):
        plt.subplot(2, 2, i+1)
        i +=1 
        plt.plot(t, y)
        plt.title(l)
        plt.xscale('log')
        if do_fit:
            mi, yf = dv.exp_fit(t, y, start_taus)
            plt.plot(t, yf)
            txt = ''
            for p in mi.params.values():
                txt += p.name + ' '
                txt += '{0:.2f}'.format(p.value) + ' \n'
            ax = plt.gca()
            plt.text(0.95, 0.95, txt, transform=ax.transAxes, va='top', ha='right')
    if fname:
        np.savetxt(fname, np.column_stack((t, pos, neg, pos/neg, total)), 
                   header = 't pos neg pos/neg total')
    
#sig_ratios(g)

def make_legend(p, err, n):
    dig = np.floor(np.log10(err))
    l = []
    for i in range(2, n + 2):
        val = str(np.around(p[i], -int(dig[i])))
        erri = str(np.ceil(round(err[i] * 10**(-dig[i]),3)) * 10**(dig[i]))
        s = ''.join(['$\\tau_', str(int(i - 1)), '=', val, '\\pm ', erri, ' $ps'])
        l.append(s)
    return l

def use_cmap(pl, cmap='RdBu', offset=0.1):
    cm = plt.get_cmap(cmap)
    idx = np.linspace(0+offset, 1-offset, len(pl))
    for i, p in enumerate(pl):
        p.set_color(cm(idx[i]))
    

def make_legend_noerr(p, err, n):
    dig = np.floor(np.log10(err))
    l = []
    for i in range(2,n+2):
        val = str(np.around(p[i], -int(dig[i])))        
        l.append('$\\tau_' + str(int(i - 1)) + '$=' + val + ' ps')
    return l
    
    
def _plot_kin_res(x):
    import networkx as nx
    res, c, A, g = fit(x[0], 'p')
    clf()
    subplot(131)
    p1=plot(wl, c[:].T)
    #plot(wl, -c[-1].T)
    xlabel('cm-1')
    ylabel('OD')
    subplot(132)
    xlabel('t')
    ylabel('conc')
    plot(f.t, A)
    xscale('log')
    subplot(133)
    

    for i in g.nodes():
        for j in g[i]:        
            g[i][j]['tau'] = '%2d'%g.edge[i][j]['tau']
            print g[i][j]['tau']

    pos = {'S1_hot':(0, 3), 'S1_warm':(0,2.3),  'S1':(0, 1.5),
           'T_hot':(1, 1.5), 'T1':(1,1), 'S0': (0,0)}
#pos = nx.spring_layout(g, pos)
    col = [i.get_color() for i in p1]
    nx.draw(g, pos, node_size=2000, node_color=col)        
    nx.draw_networkx_edge_labels(g, pos)
    figure()
    for i in [0, 5, 10, 20, -1]:
        plot(wl, f.data[i, :],'ro')
        plot(wl, (f.data - res)[i, :],'k')
        plot(wl, res[i,:])
        
if __name__ == '__main__':
    sig_ratios(f, do_fit=1, tmin=5., start_taus=[5])