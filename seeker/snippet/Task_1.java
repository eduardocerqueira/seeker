//date: 2023-04-19T17:08:52Z
//url: https://api.github.com/gists/40418abd013cb4e43c9cab54e3f56602
//owner: https://api.github.com/users/EmORz

import java.util.Scanner;

public class Task_1 {
    public static void main(String[] args) {
        /*Да се напише програма, в която потребителя
           въвежда следната информация за шоколад:
           ▪ име
           ▪ грамаж
           ▪ процент въглехидрати (дробно число от 0.1 до
           0.99)
           ▪ подходящ ли е за вегани
           ▪ цена (дробно число)
           Програмата отпечатва въведената информация в
           подходящ вид
*/
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter chocolate name: ");
        var nameChoco = sc.next();
        System.out.println("Enter chocolate weight: ");
        var weight = sc.nextInt();
        System.out.println("Enter percent of carbohydrates: ");
        var percentcarbohydrates = sc.nextDouble();
        System.out.println("Is it ok for vegan? true/false");
        var isVegan = sc.nextBoolean();
        System.out.println("Enter price: ");
        var price = sc.nextDouble();

        var print = "Chocolate name: "+nameChoco+"\nChocolate weight: "+weight+
                " gram\nPercentage of carbohydrates: "+percentcarbohydrates+
                " %\nIs ok for vegan? "+isVegan+
                "\nPrice: "+price+" lv.";

        System.out.println(print);


    }
}
