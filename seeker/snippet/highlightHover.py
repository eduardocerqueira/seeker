#date: 2021-09-29T16:55:45Z
#url: https://api.github.com/gists/c3148220a78b490d99d30b344db5915b
#owner: https://api.github.com/users/MaurizioB

from PyQt5 import QtCore, QtGui, QtWidgets
from random import randrange, choice
from string import ascii_lowercase as letters


class HighlightTextEdit(QtWidgets.QTextEdit):
    _highlightColor = QtGui.QColor('#FFFF00')
    _highlightUnderline = False
    highlightPos = -1, -1
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewport().setMouseTracking(True)
        self.highlightBlock = self.document().firstBlock()
        self.document().contentsChanged.connect(self.highlight)

    @QtCore.pyqtProperty(QtGui.QColor)
    def highlightColor(self):
        return QtGui.QColor(self._highlightColor)

    @highlightColor.setter
    def highlightColor(self, color):
        color = QtGui.QColor(color)
        if self._highlightColor != color:
            self._highlightColor = color
            self.highlight()

    @QtCore.pyqtProperty(bool)
    def highlightUnderline(self):
        return self._highlightUnderline

    @highlightUnderline.setter
    def highlightUnderline(self, underline):
        if self._highlightUnderline != underline:
            self._highlightUnderline = underline
            self.highlight()

    @property
    def highlightFormat(self):
        try:
            return self._highlightFormat
        except:
            self._highlightFormat = QtGui.QTextCharFormat()
            if self._highlightUnderline:
                self._highlightFormat.setFontUnderline(True)
            self._highlightFormat.setBackground(
                QtGui.QBrush(self._highlightColor))
            return self._highlightFormat

    def highlight(self, pos=None):
        if not self.toPlainText() or not self.isVisible():
            return
        if pos is None:
            pos = self.mapFromGlobal(QtGui.QCursor.pos())
        cursor = self.cursorForPosition(pos)
        cursor.select(cursor.WordUnderCursor)

        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        doc = self.document()
        block = doc.findBlock(start)

        # check if the mouse is actually inside the rectangle of the block
        blockRect = doc.documentLayout().blockBoundingRect(block)
        if not pos in blockRect.translated(0, -self.verticalScrollBar().value()):
            # mouse is outside of the block, no highlight
            start = end = -1

        if self.highlightPos == (start, end):
            return

        # clear the previous highlighting
        self.highlightBlock.layout().clearFormats()
        self.highlightPos = start, end
        length = end - start
        if length:
            # create a FormatRange for the highlight using the current format
            r = QtGui.QTextLayout.FormatRange()
            r.start = start - block.position()
            r.length = length
            r.format = self.highlightFormat
            block.layout().setFormats([r])

        # notify that the document must be layed out (and repainted) again
        dirtyEnd = max(
            self.highlightBlock.position() + self.highlightBlock.length(), 
            block.position() + block.length()
        )
        dirtyStart = min(self.highlightBlock.position(), block.position())
        doc.markContentsDirty(dirtyStart, dirtyEnd - dirtyStart)
        self.highlightBlock = block

    def viewportEvent(self, event):
        if event.type() == event.Leave:
            # disable highlight when leaving, using coordinates outside of the
            # viewport to ensure that highlighting is cleared
            self.highlight(QtCore.QPoint(-1, -1))
        elif event.type() == event.MouseMove:
            if not event.buttons():
                self.highlight(event.pos())
        elif event.type() == event.MouseButtonRelease:
            self.highlight(event.pos())
        return super().viewportEvent(event)


if __name__ == '__main__':
    text = ''
    for p in range(randrange(10, 30)):
        parag = []
        for w in range(randrange(5, 50)):
            word = ''
            for l in range(randrange(2, 20)):
                word += choice(letters)
            if not randrange(10):
                word += ','
            parag.append(word)
        text += ' '.join(parag).capitalize().rstrip(',') + '.\n\n'

    import sys
    app = QtWidgets.QApplication(sys.argv)
    test = HighlightTextEdit(highlightUnderline=True, 
        highlightColor=QtGui.QColor('aqua'))
    test.setText(text)
    test.show()
    sys.exit(app.exec_())