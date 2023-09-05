//date: 2023-09-05T17:03:03Z
//url: https://api.github.com/gists/00bdaa809b25a381b2180938a5e4e4a5
//owner: https://api.github.com/users/Trueno110


public class Main {
    public static void main(String[] args) {
        String printThreeWords = "1: Orange,Banana,Apple";
        System.out.println(printThreeWords);
///////////
        int a = 7;
        int b = -6;
        int checkSumSign = a+b;
        if (checkSumSign >= 0 ) {
            System.out.println("2: Сумма положительна");
        }else{
            System.out.println("2: Сумма отрицательна");
        }
///////////
        int value = 101;
        if ( value <= 0){
            System.out.println("3: Красный");
        }
        else if (value > 100){
            System.out.println("3: Зеленый");
        }
        else {
            System.out.println("3: Желтый");
        }
//////////
        int A = 7;
        int B = -6;
        if (A>=B){
            System.out.println("4: a>=b");
        }
        else {
            System.out.println("4: a<b");
        }
    }
}