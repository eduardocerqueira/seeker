//date: 2024-05-21T16:51:08Z
//url: https://api.github.com/gists/06a2af3079c7654e758daf457ab8379a
//owner: https://api.github.com/users/maulanafanny

public class Parallelogram implements Quadrilateral {
    private double base;
    private double height;
    private double side;

    public Parallelogram(double base, double height, double side) {
        this.base = base;
        this.height = height;
        this.side = side;
    }

    @Override
    public double calculateArea() {
        return base * height;
    }

    @Override
    public double calculatePerimeter() {
        return 2 * (base + side);
    }
}