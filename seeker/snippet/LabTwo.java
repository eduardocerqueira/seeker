//date: 2023-02-20T16:54:51Z
//url: https://api.github.com/gists/255a4c1f1827f4a8e568b3e4d12be228
//owner: https://api.github.com/users/jjackson813

package int2200;
/*
The main components needed to be developed in this lab is to calculate the total
cost of the paint job. I must make sure calculations are set up accurately
as I will be considering numerous amount of factors such as square feet of wall space to be painted,
price of the paint per gallon, number of gallons of paint required, hours of labor required
cost of the paint, labor charges,and total cost of the paint job.
 */
import java.util.Scanner;
import java.text.DecimalFormat;
public class LabTwo {

    public static final DecimalFormat df = new DecimalFormat("0.00");
    public static void main(String[] args) {
        Scanner reader = new Scanner(System.in); // Will allow user to input any number
        final int SquareFeet = 115;
        final int Hours = 8;
        final double Wage = 20.00;

        // Note: 115 sq ft = 8 hrs and 1 gallon of paint. $20 per hour
        double sq_foot;
        System.out.print("How much of your wall needs to be painted? ");
        sq_foot = reader.nextInt();

        if (sq_foot >= 0.0) {
            System.out.println(sq_foot + " square feet of wall space needs to be painted.");
        } else {
            System.exit(0);
        }

        double price_paint;
        System.out.print("How much is the paint per gallon? ");
        price_paint = reader.nextDouble();

        if (price_paint >= 0.0 ) {
            System.out.println("The paint per gallon costs $" + df.format(price_paint));
        } else {
            System.exit(0);
        }

        //Calculations

        // Number of gallons of paint required
        double gallons = (sq_foot/SquareFeet);
        // Hours of labor required
        double labor = (gallons * Hours);
        // Cost of the paint
        double cost_paint = (gallons * price_paint);
        // Labor charges
        double labor_charges = (labor * Wage);
        // Total cost of paint job
        double costOf_paint = (cost_paint + labor_charges);

        System.out.println(gallons + " gallons of paint required.");
        System.out.println(labor + " hours of labor is required.");
        System.out.println("The cost of paint is $" + df.format(cost_paint) + ".");
        System.out.println("You will be charged $" + df.format(labor_charges) + ".");
        System.out.println("Altogether, the paint job costs $" + df.format(costOf_paint) + "!");








    }
}
