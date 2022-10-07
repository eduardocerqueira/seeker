//date: 2022-10-07T17:27:06Z
//url: https://api.github.com/gists/aa5ec1e3aef8f80a83ae7eeb263f230d
//owner: https://api.github.com/users/7749088794

import java.io.*;
public class array {
    public static void main(String[] args)
    {
        scanner in= new scanner(System.in);
        int i= in.nextLine();
        int j= in.nextline();
        int arr=new int[i][j];
        int count=1;
        for(int m=0;m<arr.length;m++)
        {
            for(int n=0;<n<arr[m].length;n++)
            {
                arr[m][n]=count;
                count++;
            }
        }
        for(int m=0;m<arr.length;m++) {
            for (int n = 0;<n < arr[m].length;n++)
            {
                System.out.print(arr[m][n]+" ");
            }
        }


    }
}
