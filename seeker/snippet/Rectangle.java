//date: 2024-05-21T16:51:08Z
//url: https://api.github.com/gists/06a2af3079c7654e758daf457ab8379a
//owner: https://api.github.com/users/maulanafanny

public class Rectangle implements Quadrilateral {
    private double width;
    private double height;

    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public double calculateArea() {
        return width * height;
    }

    @Override
    public double calculatePerimeter() {
        return 2 * (width + height);
    }
}