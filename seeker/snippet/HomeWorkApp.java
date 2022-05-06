//date: 2022-05-06T17:20:09Z
//url: https://api.github.com/gists/5e6dc67917b90d305364cda2a3358107
//owner: https://api.github.com/users/VicBro

// Punkt 1
public class HomeWorkApp {
    public static void main(String[] args) {
        // Punkt 6 (Vyzvat Punkt 2 - Punkt 5)
        printThreeWords();
        checkSumSign();
        printColor();
        compareNumbers ();
    }

    // Punkt 2
    public static void printThreeWords() {
    System.out.println("Orange");
    System.out.println("Banana");
    System.out.println("Apple");
}

// Punkt 3
    public static void checkSumSign() {
    int a = 10;
    int b = 20;
    int sum = a + b;

    if (sum >= 0) {
        System.out.println("Сумма положительная");
        } else {
        System.out.println("Сумма отрицательная");
        }
    }

    // Punkt 4
    public static void printColor() {
        int value = 100;
if (value <= 0) {
        System.out.println("Красный");
        } else if (value > 0 & value <= 100) {
        System.out.println("Желтый");
        } else {
        System.out.println("Зелёный");
        }
        }

        // Punkt 5
public static void compareNumbers() {
        int a = 10;
        int b = 30;
        if(a >= b) {
        System.out.println("a >= b");
        } else {
        System.out.println("a < b");
        }
        }
        }