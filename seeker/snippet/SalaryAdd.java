//date: 2022-07-18T17:18:20Z
//url: https://api.github.com/gists/4ff677cdcd9a2cd2828d979bd544ed40
//owner: https://api.github.com/users/zmardil

public class SalaryAdd {
    public static void main(String[] args) {
        double salary = 100.00;
        double interest = 3.0;

        double month1 = salary / interest;
        double month2 = salary / interest;
        double month3 = salary / interest;

        double answer = (month1 + month2+ month3);

        System.out.println(month1);
        System.out.println(month2);
        System.out.println(month3);

        System.out.println(answer);
    }
}
