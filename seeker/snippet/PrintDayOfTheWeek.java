//date: 2022-04-19T16:54:23Z
//url: https://api.github.com/gists/31bfcc6ae38bf4a209ac3785650ba4e8
//owner: https://api.github.com/users/speters33w

//Udemy
//Java Programming Masterclass covering Java 11 & Java 17
//Day of the Week Challenge
//
//Write a method with the name printDayOfTheWeek that has one parameter of type int and name it day.
//
//The method should not return any value (hint: void)
//
//Using a switch statement print "Sunday", "Monday", ... ,"Saturday" if the int parameter "day" is 0, 1, ... , 6 respectively,
//otherwise it should print "Invalid day".
//
//Bonus:
// 	Write a second solution using if then else, instead of using switch.
//	Create a new project in IntelliJ with the  name "DayOfTheWeekChallenge"

import java.util.Random;
import java.util.Scanner;

/**
 * Prints the day of the week from an integer input [1-7]
 */
public class PrintDayOfTheWeek {
    /**
     * Uses switch to print the day of the week from an integer input [1-7]
     *
     * @param day The day of the week in integer format [1-7]
     */
    public static void printDayOfTheWeek(int day){
        switch(day){
            case 0: System.out.println("Sunday");break;
            case 1: System.out.println("Monday");break;
            case 2: System.out.println("Tuesday");break;
            case 3: System.out.println("Wednesday");break;
            case 4: System.out.println("Thursday");break;
            case 5: System.out.println("Friday");break;
            case 6: System.out.println("Saturday");break;
            default: System.out.println("Invalid day");break;
        }
    }

    /**
     * Uses if to print the day of the week from an integer input [1-7]
     *
     * @param day The day of the week in integer format [1-7]
     */
    public static void printDayOfTheWeekBonus(int day){
        if(day>-1 && day<7) {
            if (day == 0) System.out.println("Sunday");
            else if (day == 1) System.out.println("Monday");
            else if (day == 2) System.out.println("Tuesday");
            else if (day == 3) System.out.println("Wednesday");
            else if (day == 4) System.out.println("Thursday");
            else if (day == 5) System.out.println("Friday");
            else if (day == 6) System.out.println("Saturday");
        } else System.out.println("Invalid day");
    }

    /**
     * Generates a random day of the week between 1 and 8 and runs the program.
     */
    public static void main(String[] args) {

        //get user input to generate trial cases
        System.out.println("Press return to continue, q quits.");
        Scanner scanner = new Scanner(System.in);
        while (!"q".equals(scanner.nextLine())) {
            //Generate a random number day of the week including an invalid value[1-8]
            int day = new Random().nextInt(8) + 1;
            System.out.println(day); //prints day of week as 1-8
            printDayOfTheWeek(day-1); //sends parameter as 0-7
//            printDayOfTheWeekBonus(day-1); //sends parameter as 0-7
        }
    }
}
