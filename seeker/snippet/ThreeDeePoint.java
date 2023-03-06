//date: 2023-03-06T17:09:52Z
//url: https://api.github.com/gists/70e05e1809779173a2f4170ed9aba9d4
//owner: https://api.github.com/users/NotAName320

public class ThreeDeePoint extends Point {
    private int z;

    // public ThreeDeePoint(int x, int y, int z) {
    //     this.setX(x);
    //     this.setY(y);

    //     this.z = z;
    // }

    public ThreeDeePoint(int x, int y, int z) {
        super(x, y);

        this.z = z;
    }

    public int getZ() {
        return z;
    }

    public void setZ(int z) {
        this.z = z;
    }

    public String toString() {
        return "(" + this.getX() + ", " + this.getY() + ", " + this.getZ() + ")"; 
    }

    public double distanceFromOrigin() {
        return Math.sqrt(this.getX()*this.getX()+this.getY()*this.getY()+z*z);
    }
}
