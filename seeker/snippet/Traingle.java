//date: 2022-11-09T17:03:52Z
//url: https://api.github.com/gists/d3f17fb2647cc8dca241f6c353cb26c3
//owner: https://api.github.com/users/itsurgolu

//Java program to calculate the area of a triangle when three sides are given or normal method.
//Area = (width*height)/2
//link:- https://onlinegdb.com/KaZHwT5k_
import java.util.Scanner;
public class Triangle{
    public static void main (String[] args) {
    Scanner sc =new Scanner(System.in);
    double w,h,area;
    System.out.print("Enter the width of triangle ");
    w=sc.nextDouble();
    System.out.print("Enter the height of triangle ");
    h=sc.nextDouble();
    area=(w*h)/2;
    System.out.print("The Area Of Triangle is "+area);
    }
}