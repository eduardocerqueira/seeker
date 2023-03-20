//date: 2023-03-20T17:11:00Z
//url: https://api.github.com/gists/543fd1c159f2e710c21b31427af08a35
//owner: https://api.github.com/users/monalizadsg

import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
public class Rental {
	private Vehicle vehicle;
	private LocalDate pickUpDate;
	private LocalDate dropOffDate;
	
	public Rental(Vehicle vehicle, LocalDate pickUpDate, LocalDate dropOffDate) {
		this.vehicle = vehicle;
		this.pickUpDate = pickUpDate;
		this.dropOffDate = dropOffDate;
	}

	public double calculateTotal() {
		double duration = ChronoUnit.DAYS.between(pickUpDate, dropOffDate); 
		return duration * vehicle.getRate(); 
	}
	
	public String getDescription() {
		String result = "Rented vehicle:\n" + vehicle.getDescription() +
				"\nPick up date: " + pickUpDate + 
				"\nDrop off date: " + dropOffDate + 
				"\nRental rate: $" + calculateTotal();
		return result;
	}
}
