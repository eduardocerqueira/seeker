//date: 2024-08-26T16:49:19Z
//url: https://api.github.com/gists/b29aeaabf6b03dcde670c3c7b18786b3
//owner: https://api.github.com/users/RamshaMohammed

import java.util.Scanner;

class InvalidAgeException extends Exception {
    public InvalidAgeException() 
    {
        System.out.println("To open Bank Account using Age"); 
    }
}

class BankAccount_Registration {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);
        System.out.println("Enter your age to open a bank account:");
        int age = s.nextInt();

        try {
            validateAge(age);
            System.out.println("Congrats! Your bank account has been opened.");
        } catch (InvalidAgeException e) {
            System.out.println("Exception caught: " + e.getMessage());
        } finally {
            System.out.println("Bank account registration process completed.");
            s.close(); 
        }
    }
    public static void validateAge(int age) throws InvalidAgeException {
        if (age < 18 || age > 100) 
        {
            throw new InvalidAgeException();
       }
   }
}