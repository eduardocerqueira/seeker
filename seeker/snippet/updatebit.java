//date: 2023-04-04T17:02:00Z
//url: https://api.github.com/gists/568e78faca615e21bc096225fd98ca21
//owner: https://api.github.com/users/kumarsumiit

import java.util.Scanner;
public class updatebit {
    public static void main(String[] args) {
        Scanner sc= new Scanner(System.in);
        int oper= sc.nextInt();
        //oper=1 : set oper = 0:
        int n =5;//0101
        int pos = 1;

        int bitmask = 1<<pos;
        if(oper == 1){
            //set
            int newnumber = bitmask | n;
            System.out.println(newnumber);
        }
        else{
           // clear
            int newbitmask = ~(bitmask);
            int newnumber = newbitmask & n;
            System.out.println(newnumber);
        }
    }
}
