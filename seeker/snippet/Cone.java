//date: 2024-05-21T16:51:08Z
//url: https://api.github.com/gists/06a2af3079c7654e758daf457ab8379a
//owner: https://api.github.com/users/maulanafanny

public class Cone implements IShape3D {
    private double base;
    private double height;

    public Cone(double base, double height) {
        this.base = base;
        this.height = height;
    }

    @Override
    public double calculateVolume() {
        return (1.0 / 3) * base * height;
    }

    @Override
    public double calculateArea() {
        return 3.14 * base * (base + Math.sqrt(height * height + base * base));
    }
}