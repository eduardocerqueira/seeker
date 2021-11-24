//date: 2021-11-24T17:04:17Z
//url: https://api.github.com/gists/c857c7cc842df860dd08e94b74a421fc
//owner: https://api.github.com/users/aquaduck123


class Point {
  private double x;
  private double y;

  public Point(double x, double y) {
    this.x = x;
    this.y = y;
  }

  public Point() {
    x = 0;
    y = 0;
  }

  public String toString() {
    return "(" + x + ", " + y + ")";
  }

  public double getX() {
    return x;
  }

  public double getY() {
    return y;
  }
}

class Triangle {

  private Point p1;
  private Point p2;
  private Point p3;

  public Triangle(Point p1, Point p2, Point p3) {
    this.p1 = p1;
    this.p2 = p2;
    this.p3 = p3;
  }

  public double getLength(Point p1, Point p2) {
    double y = p2.getY() - p1.getY();
    double x = p2.getX() - p1.getX();
    return Math.sqrt((x * x) + (y * y));

  }

  public boolean isTriangle() {

    return !((getLength(p1, p2) + getLength(p2, p3) < getLength(p3, p1)));

  }

  public String typeTriangle() {
    double side1 = getLength(p1, p2);
    double side2 = getLength(p2, p3);
    double side3 = getLength(p3, p1);
    if(!isTriangle()) return "not a triangle";
    if (side2 == side3 && side3 == side1) {
      return "equaladeral";
    }


    // double p1p2 = Math.atan2(p2.getY() - p1.getY(), p2.getX() - p1.getX()) * (180.0 / Math.PI);
    // double p2p3 = Math.atan2(p3.getY() - p2.getY(), p3.getX() - p2.getX()) * (180.0 / Math.PI);
    // double p3p1 = Math.atan2(p3.getY() - p1.getY(), p3.getX() - p1.getX()) * (180.0 / Math.PI);

    if ((side1*side1)+(side2*side2) == (side3*side3)) {
      return "right";

    } else if ((side1 != side2 && side2 != side3 && side3 != side1)) {
      return "scalene";
    } else if ((side1 != side2 && side1 == side3) || (side1 == side2 && side2 != side3)
        || (side1 == side3 && side3 != side2)) {
      return "isocelese";
    }
    return "Design and implement a Point class with the following characteristics: It has two values (x &amp; y). These should both be doubles. The default location for a point is the origin (0, 0). Once the point is created you need to be able to get its x and y values. You do not need to be able to change them. You need a toString that will return a String in the form “(x, y)” Implement a small tester that creates a point,";
    
  }
}

class Main {
  public static void main(String[] args) {
    Triangle tri = new Triangle(new Point(3, 0), new Point(0, 0), new Point(0, 4));

    System.out.println(tri.typeTriangle());
  }
}