//date: 2022-02-10T16:49:31Z
//url: https://api.github.com/gists/d8db0e84aa0bb7baea4d99f74db802fb
//owner: https://api.github.com/users/VeNoM-rj

package com.company;

class Animal{
    int legs;

    public int getLegs() {
        return legs;
    }

    public void setLegs(int legs) {
        this.legs = legs;
    }

    public Animal() {
        System.out.println("I am Animal Constructor.");;
    }

    public Animal(int legs) {
        this.legs = legs;
        System.out.println("I am overloaded Animal constructor with legs value."+this.legs);
    }

    boolean isCarnivorous(){
        return true;
    }
    boolean isHerbivorous(){
        return true;
    }
    boolean canWalk(){
        return true;
    }
}

class Cat extends Animal {
    public Cat() {
        //super(3);
        System.out.println("I am Cat Constructor!");
    }

    public Cat(int legs, int hands){
        super(legs);
        //this.legs = legs;
        System.out.println("I am overloaded cat constructor with leg value: "+legs+" and hand value: "+hands);
    }


    void meowing(){
        System.out.println("Cats meow.");;
    }
    boolean canJump(){
        return true;
    }
}

class Cat1 extends Cat{
    public Cat1() {
        //super();
        System.out.println("I am Cat s child Constructor.");
    }

    public Cat1(int legs,int hands, int eyes){
        super(legs , hands);
        System.out.println("I am Overloaded cat s child constructor with leg value: "+legs+" and hand value: "+hands+" and eye value: "+eyes);
    }

}

class Dog extends Animal{

    public Dog(){
        super(3);
        System.out.println("I am a Dog Constructor!.");
    }

    void barking(){
        System.out.println("Dog Barks...");
    }

    void canRun(){
        System.out.println("Dogs can run fast.");
    }
}

/*
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
*/

public class OOPInheritance {

    public static void main(String[] args) {
        Cylinders cldr1 = new Cylinders(7,14);
        System.out.println("Lateral Surface area of Cylinder: "+cldr1.lateralSurfaceArea());
        System.out.println("Total Surface area of Cylinder: "+cldr1.totalSurfaceArea());
        System.out.println("Volume of Cylinder: "+cldr1.volume());
        //Cat cat1 = new Cat(2, 3);
        /*//cat1.setLegs(4);
        System.out.println(cat1.getLegs());
        System.out.println("Can a cat walk? "+cat1.canWalk()+" \nCan a cat jump? "+cat1.canJump());
        cat1.meowing();
        System.out.println("Do cats eat meat? "+cat1.isCarnivorous());
*/
        //
        /*Dog dog1 = new Dog();
        dog1.barking();
        dog1.canRun();
*/
        //
        //Cat1 cat11 = new Cat1(2,2,2);
    }
}
