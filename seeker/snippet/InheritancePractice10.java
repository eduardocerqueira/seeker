//date: 2022-02-10T16:49:21Z
//url: https://api.github.com/gists/0ba65ee6d6e016873957e8c5218dba3b
//owner: https://api.github.com/users/VeNoM-rj

package com.company;


class Circles{
    int r;
    private double pi = 3.14;

    public Circles(int r) {
        this.r = r;
        System.out.println("Radius: "+this.r);
    }

    public double perimeter(){
        return 2*pi*r;
    }

    public double area(){
        return pi*r*r;
    }
}

class Cylinders extends Circles {
    private int h;

    public Cylinders(int r, int h) {
        super(r);
        this.h = h;
        System.out.println("Height: "+h);
        System.out.println("YUHU i am in Practice10");
    }

    public double lateralSurfaceArea(){
        return perimeter()*h;
    }

    public double totalSurfaceArea(){
        return perimeter()*h + 2*area();
    }

    public double volume(){
        return area()*h;
    }

}


class Rectangles{
    int l;
    int b;

    public Rectangles(int l, int b) {
        this.l = l;
        this.b = b;
        System.out.println("Length: "+l+"\nBreadth: "+b);
    }

    int perimeterR(){
        return 2*(l+b);
    }
   int areaR(){
        return l*b;
   }

}

class Cuboids extends Rectangles{
    private int h;

    public Cuboids(int l, int b, int h) {
        super(l, b);
        this.h = h;
        System.out.println("Height: "+h);
    }

    int CuboidPerimeter(){
        return 2*perimeterR() + 4*h;
    }

    int cuboidArea(){
        return 2*areaR() + 2*b*h + 2*h*l;
    }

    int cuboidVolume(){
        return areaR()*h;
    }

}

public class InheritancePractice10 {
    public static void main(String[] args) {
        System.out.println("Cuboid c1: ");
        Cuboids c1 = new Cuboids(4, 6, 3);
        System.out.println("Perimeter of Cuboid: "+c1.CuboidPerimeter());
        System.out.println("Area of Cuboid: "+c1.cuboidArea());
        System.out.println("Volume of Cuboid: "+c1.cuboidVolume());

        System.out.println();

        System.out.println("Cylinder cldr1:");
        Cylinders cldr1 = new Cylinders(7,14);
        System.out.println("Lateral Surface area of Cylinder: "+cldr1.lateralSurfaceArea());
        System.out.println("Total Surface area of Cylinder: "+cldr1.totalSurfaceArea());
        System.out.println("Volume of Cylinder: "+cldr1.volume());

    }
}
