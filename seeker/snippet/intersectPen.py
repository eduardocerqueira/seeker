#date: 2024-02-07T16:58:09Z
#url: https://api.github.com/gists/37e97cdcc18a0f299c44192432a7ee8c
#owner: https://api.github.com/users/typoman

from fontTools.pens.basePen import BasePen
from fontTools.misc.bezierTools import calcCubicParameters, solveCubic, _alignment_transformation, cubicPointAtT, _line_t_of_pt, linePointAtT

def _curve_line_intersections_t(curve, line):
    aligned_curve = _alignment_transformation(line).transformPoints(curve)
    a, b, c, d = calcCubicParameters(*aligned_curve)
    intersections = solveCubic(a[1], b[1], c[1], d[1])
    return sorted(i for i in intersections if 0.0 <= i <= 1)

def curveLineIntersections(curve, line):
    """
    Finds intersections between a curve and a line.

    Args:
        curve: List of coordinates of a cubic curve as four tuples (pt1, pt2, pt3, pt4).
        line: List of coordinates of the line segment as a tuples (pt1, pt2).

    Returns:
        A list of ``Intersection`` points as tuple of (x, y).
    """

    intersections = []
    for t in _curve_line_intersections_t(curve, line):
        pt = cubicPointAtT(*curve, t)
        line_t = _line_t_of_pt(*line, pt)
        pt = linePointAtT(*line, line_t)
        intersections.append(pt)
    return intersections

def linesIntersection(line1, line2):
    """
    Calculates the intersection point of two lines.

    Parameters:
    - line1, line2: A tuple of two points (pt1, pt2), where each point is a tuple (x, y).
    """
    s1, e1 = line1
    s2, e2 = line2
    x1, y1 = s1
    x2, y2 = e1
    x3, y3 = s2
    x4, y4 = e2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom != 0:
        det_x = (x1 * y2 - y1 * x2)
        det_y = (x3 * y4 - y3 * x4)

        x = (det_x * (x3 - x4) - (x1 - x2) * det_y) / denom
        y = (det_x * (y3 - y4) - (y1 - y2) * det_y) / denom

        if (min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2) and min(x3, x4) <= x <= max(x3, x4) and min(y3, y4) <= y <= max(y3, y4)):
            return x, y


class LineIntersectPen(BasePen):
    def __init__(self, glyphSet, lineList):
        """
        Args:
        lineList: [(start_pt, end_pt), ...]
        """
        super().__init__(glyphSet)
        self.lineList = lineList
        self.intersections = set()
        self.startPt = None
        self.currentPt = None

    def _moveTo(self, pt):
        self.currentPt = pt
        self.startPt = pt

    def _lineTo(self, pt):
        for line in self.lineList:
            intersection_pt = linesIntersection((self.currentPt, pt), line)
            if intersection_pt is not None:
                self.intersections.add(intersection_pt)
        self.currentPt = pt

    def _curveToOne(self, pt1, pt2, pt3):
        for line in self.lineList:
            self.intersections.update(curveLineIntersections(
            (self.currentPt, pt1, pt2, pt3), line))
        self.currentPt = pt3

    def _closePath(self):
        if self.currentPt != self.startPt:
            self._lineTo(self.startPt)
        self.currentPt = self.startPt = None

    def getIntersections(self):
        return self.intersections


if __name__ == '__main__':
    # following is a test for a font opend by RoboFont and it's supposed to show the intersections in the drawBot extension of RF
    # first select a glyph!
    from fontTools.pens.cocoaPen import CocoaPen
    import drawBot as db
    font = CurrentFont()
    glyph = CurrentGlyph()

    def draw_pt_list(glyph, pt_list, margin=100):
        xMin, yMin, xMax, yMax = glyph.bounds
        w = xMax - xMin
        h = yMax - yMin
        db.newPage(w + (margin * 2), h + (margin * 2))
        db.translate(-xMin + margin, -yMin + margin)
        db.fill(None)
        db.stroke(0)
        db.line(start_pt, end_pt)
        db.fill(0)
        pen = CocoaPen(glyph.font)
        glyph.draw(pen)
        db.drawPath(pen.path)
        db.fill(1, 0, 0, 1)
        db.stroke(None)
        circle_radius = 5
        for point in pt_list:
            x, y = point
            db.oval(x - circle_radius, y - circle_radius,
                    circle_radius * 2, circle_radius * 2)

    y = 160 # chnage this to make the line move
    start_pt = (0, y + 100)
    end_pt = (1000, y)
    start_pt_2 = (-320, y + -80)
    end_pt_2 = (960, y)

    p = LineIntersectPen(font, [(start_pt, end_pt), (start_pt_2, end_pt_2)])
    glyph.draw(p)

    draw_pt_list(glyph, p.getIntersections())
    saveImage("test3.pdf")
