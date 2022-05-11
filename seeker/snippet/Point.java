//date: 2022-05-11T17:19:59Z
//url: https://api.github.com/gists/ca2b235fdc9e490475003f9a4037bab1
//owner: https://api.github.com/users/Deviad

package week3.assignment;

import edu.princeton.cs.algs4.StdDraw;

import java.util.Comparator;

public class Point implements Comparable<Point> {
    private final int x;
    private final int y;

    public Point(int x, int y) {
        this.x = x;
        this.y = y;
    }

    /*
     The compareTo() method should compare points by their y-coordinates,
      breaking ties by their x-coordinates. Formally, the invoking point (x0, y0) is less than the argument point
      (x1, y1) if and only if either y0 < y1 or if y0 = y1 and x0 < x1.
     */

    @Override
    public int compareTo(Point that) {
        if (this.y < that.y || this.y == that.y && this.x < that.x)
            return -1;
        else if (this.y > that.y || this.y == that.y && this.x > that.x)
            return 1;
        return 0;
    }

    public void draw() {
        /* DO NOT MODIFY */
        StdDraw.point(x, y);
    }

    public void drawTo(Point that) {
        /* DO NOT MODIFY */
        StdDraw.line(this.x, this.y, that.x, that.y);
    }

    public String toString() {
        /* DO NOT MODIFY */
        return "(" + x + ", " + y + ")";
    }

    /**
     * Returns the slope between this point and the specified point.
     * Formally, if the two points are (x0, y0) and (x1, y1), then the slope
     * is (y1 - y0) / (x1 - x0). For completeness, the slope is defined to be
     * +0.0 if the line segment connecting the two points is horizontal;
     * Double.POSITIVE_INFINITY if the line segment is vertical;
     * and Double.NEGATIVE_INFINITY if (x0, y0) and (x1, y1) are equal.
     *
     * @param that the other point
     * @return the slope between this point and the specified point
     */
    public double slopeTo(Point that) {
        if (this.x == that.x && this.y == that.y)
            return Double.NEGATIVE_INFINITY;
        else if (this.x != that.x && this.y == that.y)
            return +0.0;
        else if (this.x == that.x && this.y != that.y)
            return Double.POSITIVE_INFINITY;
        return (double) (that.y - this.y) / (that.x - this.x);
    }

    /**
     * Compares two points by the slope they make with this point.
     * The slope is defined as in the slopeTo() method.
     *
     * @return the Comparator that defines this ordering on points
     */
    public Comparator<Point> slopeOrder() {
        return new BySlope(this);
    }

    private static class BySlope implements Comparator<Point> {
        private final Point thisPoint;

        public BySlope(Point point) {
            thisPoint = point;
        }

        /*
         * The slopeOrder() method should return a comparator that compares
         * its two argument points by the slopes they make with the invoking
         * point (x0, y0). Formally, the point (x1, y1) is less than the point
         * (x2, y2) if and only if the slope (y1 − y0) / (x1 − x0) is less than
         * the slope (y2 − y0) / (x2 − x0).
         * Treat horizontal, vertical, and degenerate line segments as in the slopeTo() method.
         */
        @Override
        public int compare(Point p, Point q) {
            double slope0with1 = thisPoint.slopeTo(p);
            double slope0with2 = thisPoint.slopeTo(q);
            return Double.compare(slope0with1, slope0with2);
        }
    }
}
