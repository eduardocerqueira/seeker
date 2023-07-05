//date: 2023-07-05T16:44:56Z
//url: https://api.github.com/gists/dd6418c4c8266c55cedcbcd8f8a16c0c
//owner: https://api.github.com/users/Gundulfn

import java.util.Scanner;

public class GroceryCalculator {
    final static  double PEAR_PRICE_PRE_KG = 2.14;
    final static  double APPLE_PRICE_PRE_KG = 3.67;
    final static  double TOMATO_PRICE_PRE_KG = 1.11;
    final static  double BANANA_PRICE_PRE_KG = 0.95;
    final static  double EGGPLANT_PRICE_PRE_KG = 5;

    public static void main(String[] args) {
        double pearKg, appleKg, tomatoKg, bananaKg, eggplantKg;
        Scanner input = new Scanner(System.in);

        System.out.print("Armut Kaç Kilo ? : ");
        pearKg = input.nextDouble();
        System.out.print("Elma Kaç Kilo ? : ");
        appleKg = input.nextDouble();
        System.out.print("Domates Kaç Kilo ? : ");
        tomatoKg = input.nextDouble();
        System.out.print("Muz Kaç Kilo ? : ");
        bananaKg = input.nextDouble();
        System.out.print("Patlıcan Kaç Kilo ? : ");
        eggplantKg = input.nextDouble();

        double totalPrice = pearKg * PEAR_PRICE_PRE_KG +
                            appleKg * APPLE_PRICE_PRE_KG +
                            tomatoKg * TOMATO_PRICE_PRE_KG +
                            bananaKg * BANANA_PRICE_PRE_KG +
                            eggplantKg * EGGPLANT_PRICE_PRE_KG;

        System.out.println("Toplam Tutar : " + totalPrice + " TL");
    }
}