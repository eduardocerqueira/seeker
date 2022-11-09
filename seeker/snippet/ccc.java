//date: 2022-11-09T17:16:45Z
//url: https://api.github.com/gists/1281ef4d92079946271164c6fb961be0
//owner: https://api.github.com/users/alikendir0

import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;

public class ccc {
    public static void main(String args[]) {
        Scanner a =new Scanner(System.in);
        boolean h=true,c=true,t=true;
        System.out.println("Welcome to Real Steel Calculator!");
        System.out.println("Enter the value of hardness: ");
        double x=a.nextDouble();
        if(x>50)
        h=true;
        else h=false;
        System.out.println("Enter the value of carbon content: ");
        double y=a.nextDouble();
        if(y>7.0/10)
            c=true;
        else c=false;
        System.out.println("Enter the value of tensile strength: ");
        double z=a.nextDouble();
        if(z>5600)
            t=true;
        else t=false;

        if(h&&c&&t)
        System.out.println("This is grade 10 steel.");
        else if(h&&c&&!(t))
            System.out.println("This is grade 9 steel.");
        else if(!(h)&&c&&t)
            System.out.println("This is grade 8 steel.");
        else if(h&&!(c)&&t)
            System.out.println("This is grade 7 steel.");
        else if(h||c||t)
            System.out.println("This is grade 6 steel.");
        else System.out.println("This is grade 5 steel.");
    }
}