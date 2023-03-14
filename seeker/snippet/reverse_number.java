//date: 2023-03-14T16:48:54Z
//url: https://api.github.com/gists/2e593598c2cf5f4458d1ec7d97f6c540
//owner: https://api.github.com/users/mahesh504

package numbers;

import java.util.Scanner;

public class reverse_number {
    public static void main(String[] args)  {
        System.out.println("Enter a number");
        Scanner scanner = new Scanner(System.in);
        int digit,result=0, num = scanner.nextInt();
        while(num != 0){
            digit = num % 10;
            result = result * 10 + digit;
            System.out.println("DIGIT :"+digit);
            num = num / 10 ;
        }

        System.out.println("Result :"+result);
    }
}
