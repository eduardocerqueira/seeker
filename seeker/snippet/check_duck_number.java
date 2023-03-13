//date: 2023-03-13T17:00:17Z
//url: https://api.github.com/gists/e591bfbe4df520ef0d59563a900dff8a
//owner: https://api.github.com/users/mahesh504

package numbers;

import java.util.Scanner;

public class check_duck_number {
    public static void main(String[] args)  {
        System.out.println("Enter a number");
        Scanner scanner = new Scanner(System.in);
        int digit, num = scanner.nextInt();
        boolean isducknumber = false;

        while(num != 0){
            digit = num  % 10;
            if(digit == 0){
                isducknumber = true;
                break;
            }
            num = num / 10;

        }

       System.out.println(isducknumber ? "Duck number":"Not a duck number");

    }
}
