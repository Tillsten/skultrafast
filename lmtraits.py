# -*- coding: utf-8 -*-
"""
Created on Sat May 19 20:25:41 2012

@author: Tillsten
"""
import lmfit
from traits.api import Str, Float, Bool, HasTraits, List, Button, on_trait_change
from traitsui.api import VGroup, Item, View, TableEditor, HGroup
from traitsui.table_column import ObjectColumn
from numpy import inf

class Parameter(HasTraits):
    name=Str
    value=Float
    fixed=Bool
    max=Float(inf)
    min=Float(-inf)
    #traits_view=View(VGroup(Item('value'),Item('min_val'),
      #                      Item('max_val'),Item('fixed')))

class AllParameter(HasTraits):    
    paras=List(Parameter)
    add_para=Button('Add')
    apply_paras=Button('Apply')
    start_fit=Button('Fit')
    
    tb=TableEditor(auto_size=True)        
    cols=[ObjectColumn(name='name'),
          ObjectColumn(name='value'),
          ObjectColumn(name='fixed'),
          ObjectColumn(name='min'),
          ObjectColumn(name='max')]
    tb.columns=cols
    tb.row_factory=Parameter
    tmp=HGroup(Item('add_para', show_label=False),
               Item('apply_paras', show_label=False),
               Item('start_fit', show_label=False))
    traits_view=View(VGroup(Item('paras', editor=tb, show_label=False),
                            tmp
                           ),
                     resizable = True,
                     width = .15,
                     height = .2)
                     

    def _paras_default(self):
        p1=Parameter(name='x0',value=0.)
        p2=Parameter(name='w',value=0.1,min=0)
        p3=Parameter(name='t0',value=1.,min=0)
        return [p1,p2,p3]
    
    def _add_para_fired(self):        
        name='t'+str(len(self.paras)-2)
        p=Parameter(name=name,value=10**(len(self.paras)-2),min=0)
        self.paras.append(p)
    

        
    def from_lmparas(self,lmparams):
        self.paras=[]
        for i in lmparams:
            lm_p=lmparams[i]
            p=Parameter(name=i, value=lm_p.value, 
                        fixed=not lm_p.vary)
            if lm_p.max!=None:
                p.max=lm_p.max
            if lm_p.min!=None:
                p.min=lm_p.min
            self.paras.append(p)
            
    def to_lmparas(self):
        p=lmfit.Parameters()
        for i in self.paras:
            p.add(i.name,i.value, not i.fixed)
            if i.max!=inf:
                p[i.name].max=i.max
            if i.min!=-inf:
                p[i.name].min=i.min
        return p
        
    def to_array(self):
        return [i.value for i in self.paras]
            
            


if __name__=='__main__':
    p=lmfit.Parameters()
    p.add_many(('x0',0),('w',0.1))
    
    pa=AllParameter()
    pa.from_lmparas(p)
    pa.paras[0].value=5.
    pa.configure_traits()

