from traits.api import HasTraits, Instance, Int, List, Str, Enum, \
        on_trait_change, Any, DelegatesTo, Array,  Button, Float, Bool, Function,\
        Range, CInt

from traitsui.api import Item, View, HSplit, HGroup,\
        EnumEditor, Group, UItem, VGroup


from multiplotter import MultiPlotter
import numpy as np
class ScanViewer(HasTraits):    
    name = Str
    full_data = Array        
    _zero = Int(0)
    _num_wls = CInt
    
    wl_idx = Range(low='_zero', high='_num_wls')    
    mp = Instance(MultiPlotter)
    mean = Array
    trunc = Range('_zero', 'num_scans', value=1)            
    trunc_mean = Array    
    num_scans = Int
    channel=Range(0, 39.0)

    def __num_wls_default(self):
        print self.data.shape
        return int(self.data.shape[-2]-1)
    
    def _mp_default(self):
        mp = MultiPlotter(xaxis=range(self.data.shape[0]))
        return mp
    
    def _mean_default(self):
        return self.data.mean(-1)

    def _num_scans_default(self):
        print self.data.shape
        return int(self.data.shape[-1])

    @on_trait_change('trunc')
    def _trunc_mean_default(self):    
        tm = self.data[...,:(-self.trunc)].mean(-1)
        self.trunc_mean = tm
        return tm 

   
    @on_trait_change('channel,trunc')
    def plot(self):
        self.mp.delete_ydata()
        for i in range(self.data.shape[-1]):
            self.mp.add_ydata(str(i), self.data[:, round(10*self.channel), self.wl_idx, i],lw=1)
        self.mp.add_ydata('mean', self.mean[:, round(10*self.channel), self.wl_idx])
        self.mp.add_ydata('tr_mean', self.trunc_mean[:, round(10*self.channel), self.wl_idx], c='black')
    
    traits_view=View(VGroup(VGroup(Item('wl_idx'), Item('channel'),Item('trunc')),
                            Item('@mp', show_label = False)), resizable=True)

def apply_argsort(A, ix, ax=-1):
      ind = np.indices(A.shape)
      ind[ax] = ix
      A[ind]
      return A

def trunc_mean(d):
    idx=np.argsort(abs(d-np.median(a,-1)[...,None]),-1)
    ind = np.indices(d.shape)
    ind[-1]= idx
    return d

if __name__=='__main__':
    a = np.load('..\\br_py_ex640_para_dat10.npy')
    a = trunc_mean(a)
    sv = ScanViewer(data=a)
    sv.configure_traits()
    