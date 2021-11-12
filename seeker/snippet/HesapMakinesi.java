//date: 2021-11-12T17:03:15Z
//url: https://api.github.com/gists/4aa9be4c08212ca6121d6ee2c95a5f5b
//owner: https://api.github.com/users/mtakca

import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        int n1, n2, select;

        Scanner input = new Scanner(System.in);
        System.out.print("İlk Sayıyı Giriniz : ");
        n1 = input.nextInt();
        System.out.print("İkinci Sayıyı Giriniz : ");
        n2 = input.nextInt();

        System.out.println("1-Toplama\n2-Çıkarma\n3-Çarpma\n4-Bölme");
        System.out.print("Seçiminiz ? : ");
        select =input.nextInt();
        switch (select) {

            case 1 -> System.out.println("Toplam : " + (n1 + n2));
            case 2 -> System.out.println("Fark : " + (n1-n2));
            case 3 -> System.out.println("Çarpım : " + (n1*n2));
            case 4 -> {
                if (n2 != 0) {
                    System.out.println("Bölüm : " + (n1/n2));
                }
                else {
                    System.out.println("BİR SAYI SIFIRA BÖLÜNEMEZ!");
                }
            }
            default -> System.out.println("LÜTFEN TOPLAMA, ÇIKARMA, ÇARPMA, BÖLME ARASINDAN BİR SEÇİM YAPINIZ!");


        }
    }
}
