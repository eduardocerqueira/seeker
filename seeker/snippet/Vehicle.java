//date: 2023-05-11T17:00:55Z
//url: https://api.github.com/gists/1196e9d235c4fc35c0964f56fde30a90
//owner: https://api.github.com/users/DonMassa84


public abstract class Vehicle {
	private String licensePlate;

    public Vehicle(String licensePlate) {
        this.licensePlate = licensePlate;
    }

    public String getLicensePlate() {
        return licensePlate;
    }
}
