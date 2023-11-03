//date: 2023-11-03T16:54:25Z
//url: https://api.github.com/gists/7b50c156522eb3684d3db6df5c25dd34
//owner: https://api.github.com/users/now1j

import java.util.Scanner;
import java.util.Random;

public class MiniTest4 {
    int year;
    int month;
    int day;
    char gender;
    int random;

    public MiniTest4(int year, int month, int day, char gender, int random) {
        this.year = year;
        this.month = month;
        this.day = day;
        this.gender = gender;
        this.random = random;
    }

    public String showMiniTest4Info(){
        int Gender;
        if (gender == 'm'){
            Gender = 3;
        } else {
            Gender = 4;
        }

        int Year = year % 100;
        String formattedYear = String.format("%02d", Year);
        String formattedMonth = String.format("%02d", month);
        String formattedDay = String.format("%02d", day);
        String formattedRandom = String.format("%06d", random);

        return formattedYear + formattedMonth + formattedDay + "-" + Gender + formattedRandom;
    }

    public static void main(String[] args) {
        System.out.println("[주민등록번호 계산]");
        Scanner scanner = new Scanner(System.in);
        Random rand = new Random();

        System.out.print("출생년도를 입력해 주세요.(yyyy): ");
        int year = scanner.nextInt();
        System.out.print("출생월을 입력해 주세요.(mm): ");
        int month = scanner.nextInt();
        System.out.print("출생일을 입력해 주세요.(dd): ");
        int day = scanner.nextInt();
        System.out.print("성별을 입력해 주세요.(m/f): ");
        char gender = scanner.next().charAt(0);

        int random = 1 + rand.nextInt(999999);

        MiniTest4 person = new MiniTest4(year, month, day, gender, random);
        System.out.println(person.showMiniTest4Info());
    }

}



