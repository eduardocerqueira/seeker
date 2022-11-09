//date: 2022-11-09T17:03:52Z
//url: https://api.github.com/gists/d3f17fb2647cc8dca241f6c353cb26c3
//owner: https://api.github.com/users/itsurgolu

//Java program to calculate or to print area of a circle in a simple method
//formula area=3.14*r*r
//link: - https://onlinegdb.com/XPS8g0weS6

import java.util.Scanner;
public class Circle {
    public static void main(String args[]){
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the radius of circle ");
        //double r,area;
        double r=sc.nextDouble();
        double area= 3.14*r*r;
        System.out.print("The Area Of Circle is "+area);
        
        
        
    }
}

