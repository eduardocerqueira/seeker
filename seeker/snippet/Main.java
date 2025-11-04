//date: 2025-11-04T17:00:33Z
//url: https://api.github.com/gists/5496085d421592bde9b2868a9bc4e51d
//owner: https://api.github.com/users/nKusaka

public class Main {

    public interface Moveable {
        void move();
    }

    public class Drone implements Moveable {
        public void move() {
            System.out.println("The drone is flying");
        }
    }

    public static void main(String[] args) {
        Moveable drone = new Drone();

        drone.move();
    }
}