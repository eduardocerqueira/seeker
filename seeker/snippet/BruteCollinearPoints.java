//date: 2022-05-11T17:19:59Z
//url: https://api.github.com/gists/ca2b235fdc9e490475003f9a4037bab1
//owner: https://api.github.com/users/Deviad

package week3.assignment;

import edu.princeton.cs.algs4.Queue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.stream.Collectors;


public class BruteCollinearPoints {
    private final Point[] points;

    public BruteCollinearPoints(Point[] points) {
        if (points == null) {
            throw new IllegalArgumentException();
        }
        if (points.length < 2) {
            throw new IllegalArgumentException();
        }
        for (Point point : points) {
            if (point == null) {
                throw new IllegalArgumentException();
            }
        }
        for (int i = 0; i < points.length; i++) {
            for (int j = i + 1; j < points.length; j++) {
                if (points[i].compareTo(points[j]) == 0) {
                    throw new IllegalArgumentException(
                            "Two points are equal.");
                }
            }
        }
        this.points = Arrays.copyOf(points, points.length);
    }

    // the number of line segments
    public int numberOfSegments() {
        return segments().length;
    }

    public LineSegment[] segments() {
        if (points.length < 2) {
            throw new IllegalArgumentException();
        }
        Point[][] tmpPoints = new Point[points.length][4];
        int i = 0;
        for (Point p1 : points) {
            for (Point p2 : points) {
                for (Point p3 : points) {
                    for (Point p4 : points) {
                        if (p1.slopeTo(p2) == p2.slopeTo(p3) && p2.slopeTo(p3) == p3.slopeTo(p4) &&
                                p4.compareTo(p3) > 0 && p3.compareTo(p2) > 0 && p2.compareTo(p1) > 0) {
                            tmpPoints[i++] = new Point[]{p1, p2, p3, p4};
                        }
                    }
                }

            }
        }

        ExtendedQueue<Double> slopes = new ExtendedQueue<>();
        for (Point[] item : tmpPoints) {
            if (item != null && item.length > 0) {
                if (item[0] == null || item[1] == null) {
                    continue;
                }
                if (!slopes.contains(item[0].slopeTo(item[1]))) {
                    slopes.enqueue(item[0].slopeTo(item[1]));
                }
            }
        }
        ArrayList<ArrayList<Point>> pointsByslope = new ArrayList<>();
        int w = 0;
        for (double slope : slopes) {
            ArrayList<Point> elems = new ArrayList<>();
            pointsByslope.add(elems);

            for (Point[] pointSet : tmpPoints) {
                for (int j = 0; j < pointSet.length; j++) {
                    for (int k = j + 1; k < pointSet.length; k++) {
                        if (pointSet[j] == null || pointSet[k] == null) {
                            continue;
                        }
                        if (pointSet[j].slopeTo(pointSet[k]) == slope) {
                            if (!pointsByslope.get(w).contains(pointSet[j])) {
                                elems.add((pointSet[j]));
                            }
                            if (!pointsByslope.get(w).contains(pointSet[k])) {
                                elems.add((pointSet[k]));
                            }
                        }
                    }
                }
            }
            w++;
        }
        LineSegment[] resultArray = new LineSegment[pointsByslope.size()];
        int z = 0;
        for (ArrayList<Point> points : pointsByslope) {
            Point[] pts = points.toArray(new Point[points.size()]);
            Arrays.sort(pts);
            resultArray[z++] = new LineSegment(pts[0], pts[points.size() - 1]);
        }

        return resultArray;
    }

    private static class ExtendedQueue<T> extends Queue<T> {

        public void toArray(T[] result) {
            Iterator<T> it = this.iterator();
            int i = 0;
            while (it.hasNext()) {
                T next = it.next();
                result[i++] = next;
            }
        }

        public boolean contains(T target) {
            for (T t : this) {
                if (t.equals(target)) {
                    return true;
                }
            }
            return false;
        }

        public T get(int pos) {
            int i = 0;
            Iterator<T> it = this.iterator();
            while (it.hasNext()) {
                T next = it.next();
                if (i == pos) {
                    return next;
                }
                i++;
            }
            throw new NoSuchElementException();
        }
    }


    public static void main(String[] args) {
//        Files.readAllLines(Path.of(ClassLoader.getSystemClassLoader().getResource("points.txt").getPath()));
        Point[] points = {
                new Point(10_000, 0),
                new Point(0, 10_000),
                new Point(3_000, 7_000),
                new Point(7_000, 3_000),
                new Point(20_000, 21_000),
                new Point(3_000, 4_000),
                new Point(14_000, 15_000),
                new Point(6_000, 7_000),
        };
        Point[] points2 = {
                new Point(19_000, 10_000),
                new Point(18_000, 10_000),
                new Point(32_000, 10_000),
                new Point(21_000, 10_000),
                new Point(1234, 5678),
                new Point(14_000, 10_000),
        };
        var b = new BruteCollinearPoints(points);
        var b2 = new BruteCollinearPoints(points2);

        LineSegment[] segments = b.segments();
        LineSegment[] segments2 = b2.segments();
        assert Arrays.stream(segments).map(LineSegment::toString).collect(Collectors.joining(", ")).equals("(10000, 0) -> (0, 10000), (3000, 4000) -> (20000, 21000)");
        assert Arrays.stream(segments2).map(LineSegment::toString).collect(Collectors.joining(", ")).equals("(14000, 10000) -> (32000, 10000)");
        // draw the points
//        StdDraw.enableDoubleBuffering();
//        StdDraw.setXscale(0, 32768);
//        StdDraw.setYscale(0, 32768);
//        for (Point p : points) {
//            p.draw();
//        }
//        for (LineSegment segment : segments) {
//            StdOut.println(segment);
//            segment.draw();
//        }
//        StdDraw.show();
    }
}
