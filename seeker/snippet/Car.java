//date: 2025-11-04T17:05:23Z
//url: https://api.github.com/gists/a95630ee4cc6b663cdbb31ed05cf10ee
//owner: https://api.github.com/users/dajamiller

public interface Movable {
    void move;
    void stop;

}


public class Car implements Movable {
    private boolean isMoving;

    public Car(boolean isMoving) {
        this.isMoving = isMoving
    }

    @Override
    public void move() {
        isMoving = true;
        System.out.println("The car is moving");
    }
    
    @Override
    public void stop() {
        isMoving = false
        System.out.println("The car stopped");
}

public class Main {

    public static void main(String[] args) {

        Car car1 = new Car;

        car1.move();

    }
}

public class Drone implements Movable{}

public class Bike implements Movable{}