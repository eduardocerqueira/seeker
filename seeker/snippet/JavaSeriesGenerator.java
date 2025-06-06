//date: 2025-06-06T16:53:46Z
//url: https://api.github.com/gists/557388611e1d0eec520e268e9802025d
//owner: https://api.github.com/users/e19166

import java.util.*;
import java.io.*;

class Solution{
    public static void main(String []argh){
        Scanner in = new Scanner(System.in);
        int t=in.nextInt();
        for(int i=0;i<t;i++){
            int a = in.nextInt();
            int b = in.nextInt();
            int n = in.nextInt();
            int sum = a;
            for (int j = 0;j<n;j++){
                sum += (int) Math.pow(2,j)*b;
                System.out.print(sum+" ");
        }
            System.out.println();
            }
            
        in.close();
    }
}