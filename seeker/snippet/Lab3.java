//date: 2022-01-26T17:08:56Z
//url: https://api.github.com/gists/12b50fce4b4317b53cae1a463f7c14c6
//owner: https://api.github.com/users/DJR2904

//Deidre Rice-Pardo
//CSC 222, 1/26/22, Lab3

import java.util.Scanner;

public class Lab3 {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.print("Enter name of software product: ");  //asks user to enter product
        String p = input.next();

        String s = p.toLowerCase();    //converts input to lowercase


        switch (s) {     //switch statement for various products and their parent company

            case "word":
            case "powerpoint":
            case "notepad":
            case "excel":
            case "access":
                System.out.println("\n" + s.toUpperCase() + " is a Microsoft Product."); //returns Microsoft product
                break;

            case "docs":
            case "sheets":
            case "slides":
                System.out.println("\n" + s.toUpperCase() + " is a Google Product.");  //returns Google product
                break;

                case "textedit":
                case "pages":
                case "numbers":
                case "keynote":
                System.out.println("\n" + s.toUpperCase() + " is an Apple Product."); //returns Apple product
                break;

            default:
                System.out.println("\nError: Invalid software product"); //Gives an error message if software not listed
                System.exit(1);
        }
    }}





