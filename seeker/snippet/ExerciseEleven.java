//date: 2023-01-17T16:53:49Z
//url: https://api.github.com/gists/8c9008bc75f455b5f4ff4ebb119ee9d0
//owner: https://api.github.com/users/Punvireakroth

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

class ExerciseEleven {
    public static void main(String[] args) {
        // ask user to input a hex code and preserve it as String
        Scanner input = new Scanner(System.in);
        System.out.print("Input the HEX number: ");
        String hexNum = input.nextLine();
        System.out.println("The hex number you entered is: " + hexNum);
        // ArrayList resizable list that no need to copy old to new array
        ArrayList<String> binary = new ArrayList<String>();
        // Loop through the String
        for (int i = 0; i < hexNum.length(); i++) {
            // every character in the hex code in c varaible
            char c = hexNum.charAt(i);
            switch (c) {
                case '0':
                    binary.add("0000");
                    break;
                case '1':
                    binary.add("0001");
                    break;
                case '2':
                    binary.add("0010");
                    break;
                case '3':
                    binary.add("0011");
                    break;
                case '4':
                    binary.add("0100");
                    break;
                case '5':
                    binary.add("0101");
                    break;
                case '6':
                    binary.add("0110");
                    break;
                case '7':
                    binary.add("0111");
                    break;
                case '8':
                    binary.add("1000");
                    break;
                case '9':
                    binary.add("1001");
                    break;
                case 'A':
                case 'a':
                    binary.add("1010");
                    break;
                case 'B':
                case 'b':
                    binary.add("1011");
                    break;
                case 'C':
                case 'c':
                    binary.add("1100");
                    break;
                case 'D':
                case 'd':
                    binary.add("1101");
                    break;
                case 'E':
                case 'e':
                    binary.add("1110");
                    break;
                case 'F':
                case 'f':
                    binary.add("1111");
                    break;
                default:
                    System.out.println("Some Character in HEX number is invalid check again :)");
            }

        }

        // Display output to the console
        for(int i = 0; i < binary.size();i++){
            System.out.print(binary.get(i));
        }
    }
}
