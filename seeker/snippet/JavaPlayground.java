//date: 2025-11-04T17:07:19Z
//url: https://api.github.com/gists/30e17984ad19649a18c87d64e56090d8
//owner: https://api.github.com/users/FullStackMWilliams

public class Main {
    public static void main(String[] args) {
        Car car = new Car();
        Bike bike = new Bike();
        Drone drone = new Drone();

        car.move();
        bike.move();
        drone.move();
    }
}

// Parent class
public class Vehicle {
    void move() {
        System.out.println("Moving vehicle");
    }
}

// Child classes
public class Car extends Vehicle {
    @Override
    void move() {
        System.out.println("Car moving 🚗");
    }
}

public class Bike extends Vehicle {
    @Override
    void move() {
        System.out.println("Bike moving 🚲");
    }
}

public class Drone extends Vehicle {
    @Override
    void move() {
        System.out.println("Drone flying 🚁");
    }
}

public class Boat extends Vehicle {
    @Override
    void move() {
        System.out.println("Boat sailing ⛵");
    }
}
