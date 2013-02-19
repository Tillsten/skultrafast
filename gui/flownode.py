# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:20:58 2013

@author: tillsten
"""

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

from qtdataflow.model import Node
from qtdataflow.view import PixmapNodeView
from PyQt4.QtGui import QFileDialog, QApplication
from skultrafast.dv import tup
from skultrafast.fitter import Fitter
#from skultrafast.gui.traits_gui import MainWindow

import numpy as np

class OpenIRDat(Node):
    def __init__(self):
        super(OpenIRDat, self).__init__()
        self.node_type = 'Open IR-file'
        self.generates_output = True
        self.accepts_input = False
        self.icon_path = 'onebit_11.png'
    
    def get_view(self):
        return PixmapNodeView(self)
    
    def show_widget(self):
        fname = QFileDialog.getOpenFileName()
        a = np.loadtxt(fname)
        w = a[0, 1:]
        t = a[1:, 0]
        d = a[1:, 1:]
        self.tup = tup(w, t, d)  
    
    def get(self):
        return self.tup
        
class SvdAnalyizer(Node):
    def __init__(self):
        super(SvdAnalyizer, self).__init__()
        self.node_type = 'Open IR-file'
        self.generates_output = False
        self.accepts_input = True
        self.icon_path = 'onebit_11.png'
        
    def get_view(self):
        return PixmapNodeView(self)
    
    def show_widget(self):
        tup = self.in_conn[0].get()
        
        u, s, v = np.linalg.svd(tup.data)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.log(s))
        fig.show()
        
class ShowDat(Node):
    def __init__(self):
        super(ShowDat, self).__init__()
        self.node_type = 'Show Data'
        self.generates_output = False
        self.accepts_input = True
        self.icon_path = 'onebit_11.png'
    
    def get_view(self):
        return PixmapNodeView(self)
    
    def show_widget(self):
        tup = self.in_conn[0].get()
        #MainWindow(fitter=Fitter(tup)).edit_traits()
        
        
        
        
if __name__ == '__main__':
    
    from qtdataflow.gui import ChartWindow
    app = QApplication([])
    cw = ChartWindow()
    app.setActiveWindow(cw)
    cw.tb.add_node(OpenIRDat)
    cw.tb.add_node(ShowDat)
    cw.tb.add_node(SvdAnalyizer)
    cw.show()
    app.exec_()

