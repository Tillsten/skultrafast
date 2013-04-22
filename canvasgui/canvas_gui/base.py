from PyQt4.QtGui import *
from PyQt4.QtCore import *

class SchemaIcon(QGraphicsPixmapItem):
    def __init__(self, *args):
        super(SchemaIcon, self).__init__(*args)
        self.setAcceptHoverEvents(True)
        pixmap = QPixmap('flop.png')
        flags = [QGraphicsItem.ItemIsMovable,
                 QGraphicsItem.ItemIsSelectable]
        for f in flags:
            self.setFlag(f)


        self.setPixmap(pixmap)
        self.setScale(1.)
        self.setGraphicsEffect(None)
        self.add_label('jk')

    def add_label(self, text):
        t = QGraphicsSimpleTextItem('sjfk', self)

    def hoverEnterEvent(self, ev):
        self.setGraphicsEffect(QGraphicsColorizeEffect())

    def hoverLeaveEvent(self, ev):
        self.setGraphicsEffect(None)


class LinkLine(QGraphicsPathItem):
    def __init__(self, from_node, to_node):
        super(LinkLine, self).__init__()
        self.from_node = from_node
        self.to_node = to_node
        pen = QPen(QColor.black)
        pen.setWidth(3)
        self.setPen(pen)

    def paint(self, *args):
        start_pos = _get_right(self.from_node.sceneBoundingRect())
        end_pos = _get_left(self.to_node.sceneBoundingRect())
        path_rect = QRectF(start_pos, end_pos)
        path = QPainterPath(path_rect.topLeft())
        path.cubicTo(path_rect.topRight(),
                     path_rect.bottomLeft(),
                     path_rect.bottomRight())
        self.setPath(path)
        super(LinkLine, self).paint(*args)

def _get_right(rect):
    return QPoint(rect.right() + 5, rect.bottom() - rect.height() / 2.)

def _get_left(rect):
    return QPoint(rect.left() - 5, rect.bottom() - rect.height() / 2.)

def _get_bot(rect):
    return QPoint(rect.left() + rect.width() / 2., rect.bottom())

class SchemaNode(object):


    def __init__(self):
       pass

class Schema(QObject):
    node_connected = pyqtSignal(SchemaNode)
    node_disconnected = pyqtSignal(SchemaNode)

    def __init__(self):
        self.nodes = False


class DataInputNode(SchemaNode):
    pass

if __name__ == '__main__':
    app = QApplication([])
    sv = QGraphicsScene()
    vi = QGraphicsView()
    vi.setScene(sv)
    vi.setRenderHint(QPainter.Antialiasing)
    vi.setRenderHint(QPainter.HighQualityAntialiasing)
    it = SchemaIcon()
    it.setPos(0., 200.)
    it2 = SchemaIcon()
    sv.addItem(it)
    sv.addItem(it2)
    p = LinkLine(it, it2)
    sv.addItem(p)

    vi.show()
    app.exec_()
