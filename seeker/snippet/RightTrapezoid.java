//date: 2024-05-21T16:51:08Z
//url: https://api.github.com/gists/06a2af3079c7654e758daf457ab8379a
//owner: https://api.github.com/users/maulanafanny

public class RightTrapezoid implements Quadrilateral {
    private double base1;
    private double base2;
    private double height;
    private double side1;
    private double side2;

    public RightTrapezoid(double base1, double base2, double height, double side1, double side2) {
        this.base1 = base1;
        this.base2 = base2;
        this.height = height;
        this.side1 = side1;
        this.side2 = side2;
    }

    @Override
    public double calculateArea() {
        return ((base1 + base2) * height) / 2;
    }

    @Override
    public double calculatePerimeter() {
        return base1 + base2 + side1 + side2;
    }
}