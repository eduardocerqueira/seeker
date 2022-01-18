//date: 2022-01-18T17:06:26Z
//url: https://api.github.com/gists/a0be0bcf5e633df2dcb4e871d8cb75e3
//owner: https://api.github.com/users/Arjun2002tiwari

import java.util.ArrayList;

public class Vector1{
    public static void main(String[] args) { 

        int arr[][]={{1,2,3},{4,5,6},{7,8,9}};


        ArrayList<Integer> l1=new ArrayList<Integer>();
        ArrayList<Integer> l2=new ArrayList<Integer>();

        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                
                    if (arr[i][j] % 2 == 0) {
                        l1.add(arr[i][j]);
                    } else {
                        l2.add(arr[i][j]);
                    }
                }
        }
        System.out.println("even:");
        System.out.println(l1);
        System.out.println("odd:");
        System.out.println(l2);
    }
}