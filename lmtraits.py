# -*- coding: utf-8 -*-
"""
Created on Sat May 19 20:25:41 2012

@author: Tillsten
"""
import lmfit
from traits.api import Str, Float, Bool, HasTraits, List, Button
from traitsui.api import Group, Item, View, TableEditor
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
    apply_paras=Button('Apply')
    start_fit=Button('Fit')
    
    tb=TableEditor(auto_size=True)        
    cols=[ObjectColumn(name='name'),
          ObjectColumn(name='value'),
          ObjectColumn(name='fixed'),
          ObjectColumn(name='min'),
          ObjectColumn(name='max')]
    tb.columns=cols
    traits_view=View(Group(Item('paras', editor=tb, show_label=False),
                           ),
                     resizable = True,
                     width = .15,
                     height = .2)

    def from_lmparas(self,lmparams):
        for i in lmparams:
            lm_p=lmparams[i]
            p=Parameter(name=i, value=lm_p.value, 
                        fixed=not lm_p.vary)
            if lm_p.max:
                p.max=lm_p.max
            if lm_p.min:
                p.min=lm_p.min
            self.paras.append(p)
            
    def to_lmparas(self):
        p=lmfit.Parameters
        for i in self.paras:
            p.add(i.name,i.value, not i.fixed)
            if i.max!=inf:
                p.max=i.max
            if i.min!=-inf:
                p.min=i.min
        return p
            


if __name__=='__main__':
    p=lmfit.Parameters()
    p.add_many(('x0',0),('w',0.1))
    pa=AllParameter()
    pa.from_lmparas(p)
    pa.configure_traits()

