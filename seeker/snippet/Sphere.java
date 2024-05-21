//date: 2024-05-21T16:51:08Z
//url: https://api.github.com/gists/06a2af3079c7654e758daf457ab8379a
//owner: https://api.github.com/users/maulanafanny

public class Sphere implements IShape3D {
    private double radius;

    public Sphere(double radius) {
        this.radius = radius;
    }

    @Override
    public double calculateVolume() {
        return (4.0 / 3) * 3.14 * radius * radius * radius;
    }

    @Override
    public double calculateArea() {
        return 4 * 3.14 * radius * radius;
    }
}