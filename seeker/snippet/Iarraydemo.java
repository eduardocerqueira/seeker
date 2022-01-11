//date: 2022-01-11T17:16:03Z
//url: https://api.github.com/gists/615a1e1572ee024c4306ac5b5ad094c2
//owner: https://api.github.com/users/Pranjali4jain

import java.util.Scanner;

public class Iarraydemo
{
    static int arr[];
    public static void main(String[]args){
        arr=new int[5];
        Scanner s= new Scanner(System.in);

        for(int i=0;i< arr.length;i++)
        {
            arr[i] =s.nextInt();
        }
        for(int i=0;i< arr.length;i++)
        {
            System.out.println(arr[i]);
        }
    }
}
