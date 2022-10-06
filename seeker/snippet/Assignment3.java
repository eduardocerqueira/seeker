//date: 2022-10-06T17:11:45Z
//url: https://api.github.com/gists/67b67dc8bff68b418e88c8d421ae02b4
//owner: https://api.github.com/users/epiglottiss

/*
    채의성
    Assignment3 : Ticket Price Calculator
 */

import java.util.Scanner;
class TicketIssuer{
    public TicketIssuer() {
        if(UnitTest() == false){
            System.out.println("Warning : System Bug Exist.");
        };
    }
    public final int base = 10000;
    public final int specialDC = 4000;
    public final int normalDC = 8000;
    public int CalculatePrice(int age, int enterTime, boolean isMerit, boolean isWelfare){
        int ret = base;
        if(age<3) ret = 0;
        else if(age<13 && age>=3) ret = specialDC;
        else if(enterTime >= 17) ret = specialDC;
        else if(isMerit || isWelfare) ret = normalDC;

        return ret;
    }

    public boolean UnitTest(){
        boolean result = CalculatePrice(2,18,true,true) == 0
                && CalculatePrice(3,10,true,true) == specialDC
                && CalculatePrice(13,18,true,true) == specialDC
                && CalculatePrice(13,16,true,true) == normalDC
                && CalculatePrice(13,16,false,true) == normalDC
                && CalculatePrice(13,16,true,false) == normalDC
                && CalculatePrice(13,16,false,false) == base;
        return result;
    }
}

public class Ticket {
    public static void main(String[] args){
        System.out.println("[입장권 계산]");
        Scanner scan = new Scanner(System.in);

        System.out.print("나이 입력 (숫자) : ");
        int age = scan.nextInt();     scan.nextLine();

        System.out.print("입장시간 입력 (숫자) : ");
        int enterTime = scan.nextInt();     scan.nextLine();

        System.out.print("국가유공자? (y/n) : ");
        boolean isMerit = scan.next().equals("y");    scan.nextLine();

        System.out.print("복지카드? (y/n) : ");
        boolean isWelfare = scan.next().equals("y");  scan.nextLine();

        TicketIssuer issuer = new TicketIssuer();
        System.out.printf("입장료 : %d\n", issuer.CalculatePrice(age, enterTime,isMerit,isWelfare));
    }
}
