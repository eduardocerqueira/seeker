//date: 2024-05-21T16:51:08Z
//url: https://api.github.com/gists/06a2af3079c7654e758daf457ab8379a
//owner: https://api.github.com/users/maulanafanny

public class ShapeCalculator {

    public ShapeCalculator() {
    }

    public double calculateArea(IShape shape) {
        return shape.calculateArea();
    }

    public double calculatePerimeter(IShape2D shape) {
        return shape.calculatePerimeter();
    }

    public double calculateVolume(IShape3D shape) {
        return shape.calculateVolume();
    }
}