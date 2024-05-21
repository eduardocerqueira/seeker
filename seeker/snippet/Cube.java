//date: 2024-05-21T16:51:08Z
//url: https://api.github.com/gists/06a2af3079c7654e758daf457ab8379a
//owner: https://api.github.com/users/maulanafanny

public class Cube implements IShape3D {
    private double side;

    public Cube(double side) {
        this.side = side;
    }

    @Override
    public double calculateVolume() {
        return side * side * side;
    }

    @Override
    public double calculateArea() {
        return 6 * side * side;
    }
}