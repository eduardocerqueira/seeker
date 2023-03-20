//date: 2023-03-20T17:11:00Z
//url: https://api.github.com/gists/543fd1c159f2e710c21b31427af08a35
//owner: https://api.github.com/users/monalizadsg


public class Truck extends Vehicle {
private double cargoCapacity;
	
	public Truck(String model, double rate, double cargoCapacity) {
		super(model, rate);
		this.cargoCapacity = cargoCapacity;
	}

	public double getCargoCapacity() {
		return cargoCapacity;
	}

	@Override
	public String getDescription() {
		return "TRUCK with cargo capacity: " + cargoCapacity + "kg. " + super.getDescription();
	}	
}
