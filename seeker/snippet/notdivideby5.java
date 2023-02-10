//date: 2023-02-10T16:55:54Z
//url: https://api.github.com/gists/23f0ce6687db827badbcd0ecf3e7ca49
//owner: https://api.github.com/users/kumarsumiit

import java.util.Scanner;
public class notdivideby5 {
    public static void main(String[] args) {
        /*By using while loop and continue statement
        Print all number from 1 to n but then it should not be divisible by 5*/
        Scanner sc =new Scanner(System.in);
        int number = sc.nextInt();
        for(int i= 1; i<=number; i++){
            if ((i%5)==0){
                continue;
                //continue will escipe that number
            }
            System.out.println(i);

        }

    }
}
