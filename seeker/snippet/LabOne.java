//date: 2023-02-20T16:57:49Z
//url: https://api.github.com/gists/6d4ec5cb55e0e478f09ed2e8a8196567
//owner: https://api.github.com/users/jjackson813

package int2200;
/*
   I will develop a code that will have the user input a positive four-digit integer at the keyboard.
   The program will then display each digit of the number, one per line.
 */
import java.util.Scanner;
public class LabOne {
    public static void main(String[] args) {
        Scanner reader = new Scanner(System.in); // Will allow user to input any number
        int x; // the variable x can be any number

        System.out.println("Enter a positive 4 digit number:");
        x = reader.nextInt(); // Will help identify the number that is being contained in variable x

        System.out.println("The number you entered was " + x + // x is being printed
                ". Its digits are:"
        );
        // Using % is necessary in order to identify the remainder
        int num1 = x % 10;
        x = x / 10;
        int num2 = x % 10;
        x = x / 10;
        int num3 = x % 10;
        x = x / 10;
        int num4 = x % 10;
        x = x / 10;

        System.out.println(num4);
        System.out.println(num3);
        System.out.println(num2);
        System.out.println(num1);

        }

    }

