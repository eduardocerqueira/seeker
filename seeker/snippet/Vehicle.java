//date: 2023-03-20T17:11:00Z
//url: https://api.github.com/gists/543fd1c159f2e710c21b31427af08a35
//owner: https://api.github.com/users/monalizadsg


public abstract class Vehicle {
	public Vehicle(String model, double rate) {
		super();
		this.model = model;
		this.rate = rate;
	}

	private String model;
	private double rate;
	
	public String getModel() {
		return model;
	}
	
	public double getRate() {
		return rate;
	}
	
	public String getDescription() {
		return model + " - Daily rate: $" + rate; 
	}
}
