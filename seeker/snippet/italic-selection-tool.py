#date: 2026-01-30T17:19:15Z
#url: https://api.github.com/gists/7d6cf129e65d539db81f3539e379050b
#owner: https://api.github.com/users/adbac

# Originally written by Frederik Berlaen
# see discussion here: https://discord.com/channels/1052516637489766411/1466071066240614444

from math import radians, tan

from mojo.events import EditingTool, installTool
from mojo.UI import appearanceColorKey, getDefault


class ItalicAngleEditTool(EditingTool):
    def setup(self):
        container = self.extensionContainer(
            identifier="ItalicAngleEditTool.foreground",
            location="foreground",
            clear=True,
        )

        self.selectionFillColor = getDefault(
            appearanceColorKey("glyphViewSelectionMarqueColor")
        )
        r, g, b, a = self.selectionFillColor
        self.selectionStrokeColor = (r, g, b, 1)

        self.selectionContourLayer = container.appendPathSublayer(
            fillColor=self.selectionFillColor
        )
        self.pen = None

    def mouseDown(self, point, clickCount):
        # from RF4.6+
        # if self.mouseDownInSelection():
        self.mouseDownPoint = None
        if self._pointInSelection:
            return
        self.mouseDownPoint = point

    def mouseDragged(self, point, delta):
        if self.mouseDownPoint is None:
            return

        diffx = point.x - self.mouseDownPoint.x
        diffy = point.y - self.mouseDownPoint.y
        angle = self.getGlyph().font.info.italicAngle

        if angle is None:
            angle = 0

        angleShift = tan(radians(angle)) * diffy

        self.pen = self.selectionContourLayer.getPen(clear=True)

        self.pen.moveTo((self.mouseDownPoint.x, self.mouseDownPoint.y))
        self.pen.lineTo(
            (self.mouseDownPoint.x - angleShift, self.mouseDownPoint.y + diffy)
        )
        self.pen.lineTo(
            (
                self.mouseDownPoint.x - angleShift + (diffx + angleShift),
                self.mouseDownPoint.y + diffy,
            )
        )
        self.pen.lineTo(
            (self.mouseDownPoint.x + diffx + angleShift, self.mouseDownPoint.y)
        )
        self.pen.closePath()

    def mouseUp(self, point):
        if self.pen is None:
            return

        glyph = self.getGlyph()
        containsPoint = self.selectionContourLayer.containsPoint

        for contour in glyph:
            for point in contour.points:
                result = containsPoint((point.x, point.y))
                if result:
                    point.selected = True
                elif not self.shiftDown:
                    point.selected = False

        self.selectionContourLayer.setPath(None)

        self.pen = None

    def canSelectWithMarque(self):
        return False

    def dragSelection(self, point, delta):
        # From RF4.6+
        # if self.mouseDownInSelection():
        if self._pointInSelection:
            super().dragSelection(point, delta)

    def getToolbarTip(self):
        return "Italic Angle Edit Tool"


if __name__ == "__main__":
    installTool(ItalicAngleEditTool())
