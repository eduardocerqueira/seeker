//date: 2025-12-05T17:08:04Z
//url: https://api.github.com/gists/9f3c5e03588eb437013f8b931dc87ffc
//owner: https://api.github.com/users/DavXert

import java.util.Scanner;
public class SumaDosNumeros {
    public static void main (String[]args) {
        
        Scanner entrada = new Scanner (System.in);
        
        System.out.print ("Ingresa el primer número: ");
        int num1 = entrada.nextInt();
        System.out.print ("Ingresa el segundo número: ");
        int num2 = entrada.nextInt();
        
        Calculadora calc = new Calculadora (num1 , num2);
        
        int resuultado = calc.sumar ();
        
        System.out.println("La suma de los dos números es: " + resultado);
        
        entrada.close ();
        
    }
}