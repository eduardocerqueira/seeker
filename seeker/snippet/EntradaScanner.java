//date: 2022-09-30T17:30:43Z
//url: https://api.github.com/gists/63318bcdfc2cd2088c3a52357bed9295
//owner: https://api.github.com/users/yazmin-erazo

package EntradaScanner;

import java.util.InputMismatchException;
import java.util.Scanner;

public class EntradaScanner {
  public static void main(String[] args) {



    boolean ok = false;
    do {
      Scanner scanner = new Scanner(System.in);
      System.out.print("Ingresa dos números: ");
     
      try {
        int a = scanner.nextInt();
        int b = scanner.nextInt();
        ok = true;
      } catch (InputMismatchException e){
        System.out.println("Números invalidos");
      }
    }while (!ok);


  }
}
