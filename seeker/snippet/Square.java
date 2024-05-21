//date: 2024-05-21T16:51:08Z
//url: https://api.github.com/gists/06a2af3079c7654e758daf457ab8379a
//owner: https://api.github.com/users/maulanafanny

public class Square implements Quadrilateral {
    private double side;

    public Square(double side) {
        this.side = side;
    }

    @Override
    public double calculateArea() {
        return side * side;
    }

    @Override
    public double calculatePerimeter() {
        return 4 * side;
    }
}