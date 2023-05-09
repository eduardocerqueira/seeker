//date: 2023-05-09T16:58:43Z
//url: https://api.github.com/gists/31ac0a7007a72d6b7bcb24cf6259437f
//owner: https://api.github.com/users/m1erla

package Entrance;

import java.util.Scanner;

public class Diamond {
    public static void main(String[] args) {
        // Defined Variables
        int number;
        // Get Input From User
        Scanner input = new Scanner(System.in);
        System.out.print("Please Enter A Number :  ");
        number = input.nextInt();
        // Loop
        for (int i = 0; i <= number; i++){
            for (int j = 0; j < (number - i); j++){
                System.out.print(" ");
            }
            for (int k = 1; k <= (2 * i + 1); k++){
                System.out.print("*");
            }
            System.out.println();

        }
        // Output
        for (int i = 1; i <= number; i++){
            for (int j = 0; j < i; j++){
                System.out.print(" ");
            }
            for (int k = (2 * (number - i))+ 1; k > 0; k--){
                System.out.print("*");
            }
            System.out.println();
        }
    }
}
