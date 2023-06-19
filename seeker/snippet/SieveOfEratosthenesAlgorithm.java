//date: 2023-06-19T16:46:15Z
//url: https://api.github.com/gists/56b0813956e032d94a2023bf2f7a9ff9
//owner: https://api.github.com/users/rhnyksl

package org.javaturk.ioop.hws;

import java.util.Scanner;

public class SieveOfEratosthenesAlgorithm {
   public static void main(String[] args) {
      sieveOfEratosthenes(inputNumber());
   }
   
   private static void sieveOfEratosthenes(int inputValue) {
      boolean[] primeCorrespond = new boolean[inputValue + 1];
      for (int i = 0; i <= inputValue; i++) {
         primeCorrespond[i] = true;
      }
      
      for (int i = 2; i * i <= inputValue; i++) {
         if (primeCorrespond[i] == true) {
            for (int f = i * i; f <= inputValue; f += i) {
               primeCorrespond[f] = false;
            }
         }
      }
      
      int sumPrimeNumber = 0;
      for (int i = 2; i <= inputValue; i++) {
         if (primeCorrespond[i] == true) {
            sumPrimeNumber++;
            System.out.println(i + " ");
         }
      }
      System.out.println("Toplam asıl sayısı : " + sumPrimeNumber);
      
   }
   
   public static int inputNumber() {
      System.out.println("Bir sayı giriniz : ");
      Scanner scanner = new Scanner(System.in);
      int input = scanner.nextInt();
      return input;
   }
}

