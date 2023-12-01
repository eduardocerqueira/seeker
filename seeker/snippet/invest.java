//date: 2023-12-01T16:58:06Z
//url: https://api.github.com/gists/be28ceb8358334ed15f3153d2e160301
//owner: https://api.github.com/users/deanMoisakos

package lab5;
import java.util.Scanner;
/**
*
* @author 000832213
*/
public class Lab5 {
  /**
  * @param args the command line arguments
  */
  public static void main(String[] args) {
  int years = 1;

  Scanner input = new Scanner(System.in);

  System.out.print("The amount invested: ");

  double investmentAmount = input.nextDouble();

  System.out.print("\nAnnual interest rate: ");

  double monthlyInterestRate = input.nextDouble();

  for(int i=1;i<31;i++){
    System.out.println("Year "+i+":
    "+String.format("%.2f",futureInvestmentValue(investmentAmount,monthlyInterestRate,years)));
    years++;
    }
}
                       
  public static double futureInvestmentValue(double investmentAmount,double
  monthlyInterestRate, int years){
    double InvestmentValue;
     //double numberOfYears=0;
    InvestmentValue = (investmentAmount * (Math.pow((1+((monthlyInterestRate/100)/12)),(years*12))));
    return InvestmentValue;
  }
}