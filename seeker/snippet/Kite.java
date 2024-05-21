//date: 2024-05-21T16:51:08Z
//url: https://api.github.com/gists/06a2af3079c7654e758daf457ab8379a
//owner: https://api.github.com/users/maulanafanny

public class Kite implements Quadrilateral {
    private double diagonal1;
    private double diagonal2;
    private double side1;
    private double side2;

    public Kite(double diagonal1, double diagonal2, double side1, double side2) {
        this.diagonal1 = diagonal1;
        this.diagonal2 = diagonal2;
        this.side1 = side1;
        this.side2 = side2;
    }

    @Override
    public double calculateArea() {
        return (diagonal1 * diagonal2) / 2;
    }

    @Override
    public double calculatePerimeter() {
        return 2 * (side1 + side2);
    }
}