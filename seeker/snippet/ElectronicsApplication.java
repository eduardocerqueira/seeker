//date: 2023-10-23T17:00:22Z
//url: https://api.github.com/gists/0840dc3dc4518567b83967e08f95a3ae
//owner: https://api.github.com/users/AaronRubinos

package AdapterPattern;

import java.util.Scanner;

public class ElectronicsApplication {
    public static void main(String[] args){

        Scanner scanner = new Scanner(System.in);

        // get device.
        Laptop laptop = new Laptop();
        Refrigerator ref = new Refrigerator();
        SmartphoneCharger charger = new SmartphoneCharger();

        while (true) {
            System.out.println("Smart Electronic Devices Menu:");
            System.out.println("1. Charge Laptop");
            System.out.println("2. Start Refrigerator");
            System.out.println("3. Charge Smartphone");
            System.out.println("4. Exit");

            // to choose user what electronic device she/he wants to plug in.
            System.out.print("Select an option: ");
            int userChoice = getUserChoice(scanner);
            System.out.println();

            switch (userChoice) {
                case 1: // for laptop.
                    PowerOutlet laptopOutlet = new LaptopAdapter(laptop);
                    laptopOutlet.plugIn();
                    break;

                case 2: // for refrigerator.
                    PowerOutlet refrigeratorOutlet = new RefrigeratorAdapter(ref);
                    refrigeratorOutlet.plugIn();
                    break;

                case 3: // for smartphone charger.
                    PowerOutlet smartphoneChargerOutlet = new SmartphoneAdapter(charger);
                    smartphoneChargerOutlet.plugIn();
                    break;
                case 4: // to exit the program
                    System.out.println("Exiting the program. Goodbye!");
                    System.exit(0);
                    break;
                default: // for user input invalid choice.
                    System.out.println("Invalid choice. Please select a valid option (1-4).");
                    break;
            }
        }
    }

    private static int getUserChoice(Scanner scanner) {
        while (true) {
            try {
                int choice = Integer.parseInt(scanner.nextLine());
                if (choice >= 1 && choice <= 4) {
                    return choice;
                }
                System.out.print("Invalid input. Please enter a valid option (1-4): ");
            } catch (NumberFormatException e) {
                System.out.print("Invalid input. Please enter a valid option (1-4): ");
            }
        }
    }

}