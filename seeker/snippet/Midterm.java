//date: 2023-03-20T17:11:00Z
//url: https://api.github.com/gists/543fd1c159f2e710c21b31427af08a35
//owner: https://api.github.com/users/monalizadsg

import java.time.LocalDate;
import java.util.Scanner;
import java.nio.file.*;
import java.io.*;
import static java.nio.file.StandardOpenOption.*;
public class Midterm {

	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		Vehicle vehicles[] = null;
		
		// display options
		System.out.println("Please choose from the following options:");
		System.out.println("1 - Enter Vehicles");
		System.out.println("2 - Load Vehicles");
		
		//create file path
		Path file = Paths.get("src\\" + "NewVehicles.txt");

		int option = input.nextInt();
		
		if(option == 1) {
			vehicles = writeVehiclesToFile(file); // get vehicles array from user input
		}
		else if(option == 2) {
			// ask user for fileName
			input.nextLine(); // read next line after nextInt() when choosing option 2
			System.out.println("Please enter file name: ");
			String fileName = input.nextLine();
			file = Paths.get("src\\" + fileName);
			vehicles = readVehiclesFromFile(file); // get vehicles array from file
		}

		// Select vehicle
		int selection = 0;
		System.out.println("Available vehicles:");
		for(int i = 0; i < vehicles.length; i++) {
			System.out.println((i + 1) + " " + vehicles[i].getDescription());
		}

		boolean success = false;
		while (!success) {
			try {
				System.out.println("Pick a vehicle by entering its number: ");	
				selection = input.nextInt();
				if(selection >=1 && selection <= vehicles.length) {
					success = true;
				} else {
					System.out.println("Please enter a valid number between 1 and " + vehicles.length);
				}
			} catch (Exception e) {
				System.out.println("Please enter a valid number between 1 and " + vehicles.length);
				System.out.println(e);
			}
		}

		// Enter pick up date
		System.out.println("Please enter the pick up date: ");
		LocalDate pickUpDate = inputDate();

		// Enter drop off date
		System.out.println("Please enter the drop off date: ");
		LocalDate dropOffDate = inputDate();

		// Create Rental
		Rental rental = new Rental(vehicles[selection-1], pickUpDate, dropOffDate);

		// Output rental details and price
		System.out.println(rental.getDescription());
	}

	private static Vehicle[] writeVehiclesToFile(Path file) {
		Scanner input = new Scanner(System.in);
		Vehicle[] vehicles = null;

		String model;
		double rate, cargoCapacity;
		Car.Size size;

		String s = "";
		String delimiter = "\n";

		try {
			OutputStream output = new BufferedOutputStream(Files.newOutputStream(file, CREATE));
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(output));

			System.out.println("How many vehicles do you want to create?");
			int vehicleCount = input.nextInt();

			// create vehicles array
			vehicles = new Vehicle[vehicleCount];

			// write vehicle count to file
			s = s + vehicleCount;
			writer.write(s);
			writer.newLine();

			int vehicleNumCreated = 0;
			while(vehicleNumCreated != vehicleCount)	 {
				System.out.println("Enter 1 for Car and 2 for Truck: ");
				int vehicleTypeOption = input.nextInt();
				input.nextLine();
				
				System.out.println("Enter model: ");
				model = input.nextLine();

				System.out.println("Enter rate: ");
				rate = input.nextDouble();

				// car option
				if(vehicleTypeOption == 1) {
					System.out.println("What is the size of the car?");
					for(int i = 0; i < Car.Size.values().length; i++) {
						System.out.println((i+1) + " - " + Car.Size.values()[i]);
					}
					System.out.println("Please enter your choice: ");
					int carSizeSelection = input.nextInt();
					size = Car.Size.values()[carSizeSelection - 1];

					// add vehicle to array
					vehicles[vehicleNumCreated] = new Car(model, rate, size);
				}
				// truck option
				else if(vehicleTypeOption == 2) {
					System.out.println("Please enter the cargo capacity: ");
					cargoCapacity = input.nextDouble();

					// add vehicle to array
					vehicles[vehicleNumCreated] = new Truck(model, rate, cargoCapacity);
				}

				// increment count to update vehicle created and use as the index 
				// of vehicle array when adding a new one
				vehicleNumCreated++;
			}

			// write to file
			for(int i = 0; i < vehicles.length; i++) {
				if(vehicles[i] instanceof Car) {
					Car car = (Car) vehicles[i];
					s = "Car" + delimiter + car.getModel()
					+ delimiter  + car.getRate() + delimiter + car.getSize();
				}
				else if(vehicles[i] instanceof Truck) {
					Truck truck = (Truck) vehicles[i];
					s = "Truck" + delimiter + truck.getModel()
					+ delimiter  + truck.getRate() + delimiter + truck.getCargoCapacity();
				}
				writer.write(s);
				writer.newLine();
			}
			System.out.println("Successfully written " + vehicleCount + " vehicles");
			writer.close();
		} 
		catch(Exception e)
		{
			System.out.println("Message: " + e);
		}
		
		return vehicles;
	}


	private static Vehicle[] readVehiclesFromFile(Path file) {
		Vehicle[] vehicles = null;

		try {
			InputStream fileInput = new BufferedInputStream(Files.newInputStream(file));
			BufferedReader reader = new BufferedReader(new InputStreamReader(fileInput));

			int vehicleCount = Integer.parseInt(reader.readLine()); // read first line as the vehicle count

			// create vehicles array
			vehicles = new Vehicle[vehicleCount];

			String model;
			double rate, cargoCapacity;
			Car.Size size;

			for(int i = 0; i < vehicleCount; i++) {
				String strLine = reader.readLine();

				if(strLine.equals("Car")) {
					model = reader.readLine();
					rate = Double.parseDouble(reader.readLine());
					size = Car.Size.valueOf(reader.readLine());
					vehicles[i] = new Car(model, rate, size);
				}
				else if(strLine.equals("Truck")) {
					model = reader.readLine();
					rate = Double.parseDouble(reader.readLine());
					cargoCapacity = Double.parseDouble(reader.readLine());
					vehicles[i] = new Truck(model, rate, cargoCapacity);
				}
			}
		}
		catch(Exception e)
		{
			System.out.println("Message: " + e);
		}

		return vehicles;
	}


	private static LocalDate inputDate() {
		int year, month, day;
		LocalDate date= null;
		boolean success = false;

		while (!success) {
			try {
				Scanner inputDevice = new Scanner(System.in);						
				System.out.println("Please enter year: ");
				year = inputDevice.nextInt();
				System.out.println("Please enter month: ");
				month = inputDevice.nextInt();
				System.out.println("Please enter day: ");
				day = inputDevice.nextInt();
				date = LocalDate.of(year, month, day);
				success = true;

			} catch (Exception e) {
				System.out.println("Please enter a valid date!");
				System.out.println(e);
			}
		}

		return date;
	}

}
