//date: 2023-01-06T16:49:08Z
//url: https://api.github.com/gists/568649fe0b028913ff2a295c0e51eed6
//owner: https://api.github.com/users/trsoo24

import java.util.Scanner;

public class amusementPark {

    public static void main(String[] args) {
        int pay = 10000;

        System.out.println("[입장권 계산]");
        System.out.print("나이를 입력해 주세요.(숫자입력):");

        Scanner scanner1 = new Scanner(System.in);
        int age = scanner1.nextInt();
        if(age < 3){
            pay = 0;
        }else if(age > 3 && age < 13){
            pay = pay - 4000;
        }else {
            pay = pay;
        }
        System.out.print("입장시간을 입력해 주세요.(숫자입력):");
        Scanner scanner2 = new Scanner(System.in);
        int hour = scanner2.nextInt();
        if (hour > 17 && hour < 24){
            pay = pay - 4000;
        }else {
            pay = pay;
        }
        System.out.print("국가유공자 여부를 입력해 주세요.(y/n):");
        Scanner scanner3 = new Scanner(System.in);
        String hero = scanner3.next();
        if (hero == "y"){
            pay = pay - 2000;
        } else if (hero == "n") {
            pay = pay;
        }
        System.out.print("복지카드 여부를 입력해 주세요.(y/n):");
        Scanner scanner4 = new Scanner(System.in);
        String card = scanner4.next();
        if (card == "y"){
            pay = pay - 2000;
        } else if (card == "n") {
            pay = pay;
        }
        System.out.print("입장료: " + pay);