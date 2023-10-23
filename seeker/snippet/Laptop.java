//date: 2023-10-23T17:00:22Z
//url: https://api.github.com/gists/0840dc3dc4518567b83967e08f95a3ae
//owner: https://api.github.com/users/AaronRubinos

package AdapterPattern;

public class Laptop {

    private boolean charging = false;

    public void plugIn() {
        System.out.println("Laptop is plugged in.");
        charging = true;
    }

    public boolean isCharging() {
        return charging;
    }

    public void charge() {
        if (isCharging()) {
            System.out.println("Laptop is already charging.");
            System.out.println();
        } else {
            plugIn();
            System.out.println("Laptop is now charging!");
            System.out.println();
        }
    }
}
