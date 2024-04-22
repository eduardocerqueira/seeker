//date: 2024-04-22T16:51:44Z
//url: https://api.github.com/gists/8878f438856e8897d875017a494f55c0
//owner: https://api.github.com/users/Aishwarya2233

import java.util.Scanner;
public class AreaR {
    public static void main(String[] args){
        System.out.println("Enter length and breadth as integers : ");
        Scanner sc = new Scanner(System.in);
        int length = sc.nextInt();
        int breadth = sc.nextInt();
        int area = (length * breadth);
        System.out.println(area);
    }
}

