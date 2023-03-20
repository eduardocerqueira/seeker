//date: 2023-03-20T17:11:00Z
//url: https://api.github.com/gists/543fd1c159f2e710c21b31427af08a35
//owner: https://api.github.com/users/monalizadsg


public class Car extends Vehicle {
	enum Size { COMPACT, MIDSIZE, FULLSIZE, PREMIUM };

	private Size size;
	
	public Car(String model, double rate, Size size) {
		super(model, rate);
		this.size = size;
	}
	
	public Size getSize() {
		return size;
	}

	@Override
	public String getDescription() {
		return size + " " + super.getDescription();
	}
}
