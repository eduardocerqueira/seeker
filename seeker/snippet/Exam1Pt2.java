//date: 2023-02-20T16:55:13Z
//url: https://api.github.com/gists/f00b33f5835ab8a1bc2cc1ea119c2dae
//owner: https://api.github.com/users/jjackson813

package int2200;

import java.util.Scanner;

/*
Develop a Java program to compute and report the area and the circumference of a circle, given its radius.
Have the user enter a value for RADIUS, a positive double value. Then using the value of 3.14 as PI (set up as a named constant),
calculate and display the circle's area = (pi * radius) and the circumference = (2 * PI * RADIUS).
 */
public class Exam1Pt2 {
    public static void main(String[] args) {
        Scanner reader = new Scanner(System.in);
        final double pi = 3.14; // Set up as a constant

        double x;
        System.out.print("Enter a value for RADIUS: ");
        x = reader.nextDouble();

        // Will stop when a negative value is entered
        if (x < 0.0) {
            System.exit(0);
        }

        // Calculations
        // area = pi * radius
        // circumference = 2 * pi * radius

        double circle_area = (pi * x);
        double circumference = ( 2 * pi * x);

        // Output input by user
        System.out.println("The circle's area is " + circle_area);
        System.out.println("The circumference is " + circumference);

    }
}
