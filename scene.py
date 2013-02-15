# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:19:05 2013

@author: tillsten
"""

import sys, math
from PyQt4 import QtGui, QtCore
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:44:12 2013

@author: tillsten
"""
from PyQt4.QtGui import *


class SchemaIcon(QGraphicsPixmapItem):    
    def __init__(self, *args):
        super(SchemaIcon, self).__init__(*args)
        self.setAcceptHoverEvents(True)                
        pixmap = QPixmap('flop.png')
        self.setPixmap(pixmap)
        self.setScale(0.5)
        self.setGraphicsEffect(QGraphicsDropShadowEffect())        
    
    def hoverEnterEvent(self, ev):
        self.setGraphicsEffect(QGraphicsColorizeEffect())
    
    def hoverLeaveEvent(self, ev):
        print 'yep'
        self.setGraphicsEffect(QGraphicsDropShadowEffect())


class CompartmentWindow(QtGui.QWidget):
    def __init__(self, parent=None):
        super(CompartmentWindow, self).__init__(parent)
        layout = QtGui.QHBoxLayout()
        layout.addWidget(ButtonList(self))

        self.view = MyView(self)
        layout.addWidget(self.view)

        self.setLayout(layout)

    def add_compartment(self):
        self.view.add_compartment()


class ButtonList(QtGui.QWidget):
    def __init__(self, parent):
        super(ButtonList, self).__init__(parent)
        layout = QtGui.QVBoxLayout()
        add_button = QtGui.QPushButton("Add Compartment")
        add_button.clicked.connect(parent.add_compartment)
        
        layout.addWidget(add_button)
        self.setLayout(layout)


class MyView(QtGui.QGraphicsView):
    def __init__(self, parent):
        QtGui.QGraphicsView.__init__(self, parent)
        
        self.setGeometry(QtCore.QRect(100, 100, 250, 250))
        self.scene = QtGui.QGraphicsScene(self)
        self.scene.setSceneRect(QtCore.QRectF(0, 0, 200, 200))
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self.setScene(self.scene)
        self.compartments = []
        self.transitions = []

    def add_compartment(self):
        item = AbstractItem(0., 0., 20., 20.)
        self.compartments.append(item)
        self.scene.addItem(item)

    def mousePressEvent(self, ev, **kwargs):
        super(MyView, self).mousePressEvent(ev, **kwargs)
        if ev.button() == 2:
            try:
                sel_item = self.scene.selectedItems()[0]
            except IndexError:
                sel_item = None

            at_item = self.scene.itemAt(ev.posF())


            cond = [sel_item, at_item, sel_item != at_item]
            if all(cond):

                if isinstance(at_item.parentItem(), AbstractItem):
                    at_item = at_item.parentItem()

                self.connect_compartments(sel_item, at_item)

    def connect_compartments(self, start_comp, end_comp):
        arrow = Arrow(start_comp, end_comp)

        self.transitions.append((start_comp, end_comp, arrow))
        start_comp.arrows.append(arrow)
        end_comp.arrows.append(arrow)
        self.scene.addItem(arrow)


class Arrow(QtGui.QGraphicsLineItem):
    def __init__(self, startItem, endItem, parent=None, scene=None):
        super(Arrow, self).__init__(parent, scene)

        self.arrowHead = QtGui.QPolygonF()

        self.myStartItem = startItem
        self.myEndItem = endItem
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)
        self.myColor = QtCore.Qt.black
        self.setPen(QtGui.QPen(self.myColor, 2, QtCore.Qt.SolidLine,
                               QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))

    def setColor(self, color):
        self.myColor = color

    def startItem(self):
        return self.myStartItem

    def endItem(self):
        return self.myEndItem

    def boundingRect(self):
        extra = (self.pen().width() + 20) / 2.0
        p1 = self.line().p1()
        p2 = self.line().p2()
        return QtCore.QRectF(p1, QtCore.QSizeF(p2.x() - p1.x(),
                                               p2.y() - p1.y())).normalized().adjusted(

            -extra, -extra, extra, extra)

    def shape(self):
        path = super(Arrow, self).shape()
        path.addPolygon(self.arrowHead)
        return path

    def updatePosition(self):
        pass
#        
#        p1 = self.mapFromItem(self.myStartItem, self.myStartItem.boundingRect())
#        
#        p1, p2 = facing_sides(p1, p2)
#                              
#        self.setLine(QtCore.QLineF(p1, p2))
        

    def paint(self, painter, option, widget=None):
        if (self.myStartItem.collidesWithItem(self.myEndItem)):
            return

        myStartItem = self.myStartItem
        myEndItem = self.myEndItem
        myColor = self.myColor
        myPen = self.pen()
        myPen.setColor(self.myColor)
        arrowSize = 5.0
        painter.setPen(myPen)
        painter.setBrush(self.myColor)

        #centerLine = QtCore.QLineF(myStartItem.pos(), myEndItem.pos())
        p1, p2 = facing_sides(myStartItem.sceneBoundingRect(),
                              myEndItem.sceneBoundingRect())
        self.setLine(QtCore.QLineF(p1, p2))
        line = self.line()

        angle = math.acos(line.dx() / line.length())
        if line.dy() >= 0:
            angle = (math.pi * 2.0) - angle

        arrowP1 = line.p1() + QtCore.QPointF(
            math.sin(angle + math.pi / 3.0) * arrowSize,
            math.cos(angle + math.pi / 3.0) * arrowSize)
        arrowP2 = line.p1() + QtCore.QPointF(
            math.sin(angle + math.pi - math.pi / 3.0) * arrowSize,
            math.cos(angle + math.pi - math.pi / 3.0) * arrowSize)

        self.arrowHead.clear()
        for point in [line.p1(), arrowP1, arrowP2]:
            self.arrowHead.append(point)

        painter.drawLine(line)
        painter.drawPolygon(self.arrowHead)
        if self.isSelected():
            painter.setPen(QtGui.QPen(myColor, 1, QtCore.Qt.DashLine))
            myLine = QtCore.QLineF(line)
            myLine.translate(0, 4.0)
            painter.drawLine(myLine)
            myLine.translate(0, -8.0)
            painter.drawLine(myLine)
            
def facing_sides(rect_a, rect_b):
    
    if rect_a.left() < rect_b.left():    
        left_rect = rect_a
        right_rect = rect_b
        switch = True
    else:
        left_rect = rect_b
        right_rect = rect_a
        switch = False
        
    left_point = left_rect.right() + 5, (left_rect.top() + left_rect.bottom())/2.
    right_point = right_rect.left() - 5, (right_rect.top() + right_rect.bottom())/2.
    if switch:
        left_point, right_point = right_point, left_point
    return QtCore.QPointF(*left_point), QtCore.QPointF(*right_point)
    
class AbstractItem(QtGui.QGraphicsRectItem):
    def __init__(self, *args):
        super(AbstractItem, self).__init__(*args)
        self.text_item = QtGui.QGraphicsSimpleTextItem('sf', parent=self)
        self.text_item.setPos(self.mapFromParent(self.x(), self.y()))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.text_item.setFont(font)
        self.setFlag(self.ItemIsMovable)
        self.setFlag(self.ItemIsSelectable)
        self.setFlag(self.ItemSendsGeometryChanges)
        self.setRect(self.text_item.boundingRect())
        self.arrows = []

    def mouseDoubleClickEvent(self, event):
        text, ok = QtGui.QInputDialog.getText(None, 'Input Dialog',
            'Enter name:')
        if ok and text != '':
            self.text_item.setText(text)
            self.setRect(self.text_item.boundingRect())

    def itemChange(self, change, val):
        if change == self.ItemPositionHasChanged:
            for arr in self.arrows:
                pass
                # arr.updatePosition()
        return QtGui.QGraphicsItem.itemChange(self, change, val)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    view = CompartmentWindow()
    view.add_compartment()
    view.show()
    sys.exit(app.exec_())
