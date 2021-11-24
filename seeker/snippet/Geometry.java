//date: 2021-11-24T17:03:29Z
//url: https://api.github.com/gists/13242adf9f2f23f62b65987b4abaf911
//owner: https://api.github.com/users/migmoog

class Main 
{
	public static void main(String[] args) 
	{
		// Point test
		Point pointTest = new Point(0.045, 0.86);
		System.out.println(pointTest);

		// Triangle test
		// equilateral
		Triangle e = new Triangle(new Point(2, 0), new Point(0, 0), new Point(1, 1.732));
		System.out.println(e);
		System.out.println(e.isTriangle());
		System.out.println(e.typeTriangle());
		System.out.println();

		// iscoceles
		Triangle i = new Triangle(new Point(5, 0), new Point(), new Point(2.5, 9.682));
		System.out.println(i);
		System.out.println(i.isTriangle());
		System.out.println(i.typeTriangle());
		System.out.println();

		// right
		Triangle r = new Triangle(new Point(3, 0), new Point(), new Point(0, 4));
		System.out.println(r);
		System.out.println(r.isTriangle());
		System.out.println(r.typeTriangle());
		System.out.println();

		//scalene
		Triangle s = new Triangle(new Point(2.41, 0), new Point(), new Point(4.524, 2.129));
		System.out.println(s);
		System.out.println(s.isTriangle());
		System.out.println(s.typeTriangle());
		System.out.println();

		// not a triangle
		Triangle n = new Triangle(new Point(), new Point(), new Point());
		System.out.println(n);
		System.out.println(n.isTriangle());
		System.out.println(n.typeTriangle());
		System.out.println();
	}
}

class Triangle 
{
	private Point p1;
	private Point p2;
	private Point p3;

	public Triangle(Point p1, Point p2, Point p3) 
	{
		this.p1 = p1;
		this.p2 = p2;
		this.p3 = p3;
	}

	public boolean isTriangle() 
	{
		return (sideLength(1) + sideLength(2)) > sideLength(3);
	}

	public double sideLength(int side) 
	{
		switch (side) 
		{
			case 1:
				return distBetweenPoints(p1, p2);
			case 2:
				return distBetweenPoints(p2, p3);
			case 3:
				return distBetweenPoints(p3, p1);
			default:
				return -1.0;
		}
	}

	private double distBetweenPoints(Point p1, Point p2) 
	{
		double x = p2.getX() - p1.getX();
		double y = p2.getY() - p1.getY();

		double dist = Math.sqrt((x * x) + (y * y));

		return dist;
	}

	public String typeTriangle() {
		double first = sideLength(1);
		double second = sideLength(2);
		double third = sideLength(3);

		double EPSILON = 0.1;

		boolean firstToSecond = Math.abs(first - second) < EPSILON;
		boolean firstToThird = Math.abs(first - third) < EPSILON;
		boolean secondToThird = Math.abs(second - third) < EPSILON;

		boolean pythagTo1 = ((third*third) + (second*second)) == (first*first);
		boolean pythagTo2 = ((third*third) + (first*first)) == (second*second);
		boolean pythagTo3 = ((first*first) + (second*second)) == (third*third);

		if (!isTriangle()) {
			return "not a triangle";
		} else if (firstToSecond && firstToThird) {
			return "equilateral";
		} else if (firstToSecond || firstToThird || secondToThird) {
			return "isoceles";
		} else if (pythagTo1 || pythagTo2 || pythagTo3) {
			return "right";
		} else {
			return "scalene";
		}
	}

	public String toString() {
		return p1.toString() + " " + p2.toString() + " " + p3.toString();
	}
}

class Point 
{
	private double x;
	private double y;

	public Point(double x, double y) 
	{
		this.x = x;
		this.y = y;
	}

	public Point() 
	{
		this.x = 0;
		this.y = 0;
	}

	public double getX() {return x;}

	public double getY() {return y;}

	public String toString() 
	{
		return "(" + x + "," + y + ")";
	}
}