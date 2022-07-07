//date: 2022-07-07T17:14:00Z
//url: https://api.github.com/gists/5107cbc0fa3ba2fb4532425f499e2340
//owner: https://api.github.com/users/eliya1452

import java.util.Scanner;

public class GoldenRatio {
    public static void main(String[] args) {


        Scanner scanner = new Scanner(System.in);
        System.out.println("give me three num and i will give you the avg of them");
        Double n1, n2, n3, n4;
        n1= scanner.nextDouble();
        n2= scanner.nextDouble();
        n3= scanner.nextDouble();
       n4=(n1+n2+n3)/3;
       System.out.println("avg =  " + n4);

    }
}


