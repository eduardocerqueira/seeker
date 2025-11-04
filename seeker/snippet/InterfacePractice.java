//date: 2025-11-04T17:01:59Z
//url: https://api.github.com/gists/0445c668e8cd20cc3c4f6a6c4112fbd8
//owner: https://api.github.com/users/maryjanekw

public class Main {
    public static void main(String[] args) {
        
        Moveable car = new Car();
        Moveable bike = new Bike();
        Moveable drone = new Drone();

        car.speed();
        bike.speed();
        drone.speed();
        

    }
}

interface Moveable {
    void speed();
}

class Car implements Moveable {
    public void speed() {
        System.out.println("70 mph");
    
    }
}

class Bike implements Moveable {
    public void speed(){
        System.out.println("25 mph");

    }
}

class Drone implements Moveable{
    public void speed(){
        System.out.println("20 mph");
    }
}