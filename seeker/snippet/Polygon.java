//date: 2023-02-06T16:43:27Z
//url: https://api.github.com/gists/b457e40a6b43f9351e22a478e0c8f278
//owner: https://api.github.com/users/Tgtg29

public class Polygon {
    private int numSides;
    private double sideLength;
    private String shapeType;

    public void Polygon(){
        numSides = 3;
        sideLength = 1.0;
        shapeType = "Triangle";
    }

    public void Polygon(int s, int sL, String sT){
        numSides = s;
        sideLength = sL;
        shapeType = sT;
    }

    public double getSideLength() {
        return sideLength;
    }

    public int getNumSides(){
        return numSides;
    }

    public String getShapeType(){
        return shapeType;
    }

}