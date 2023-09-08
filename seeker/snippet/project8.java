//date: 2023-09-08T16:44:46Z
//url: https://api.github.com/gists/76129fd14cdd8d8543b45177ba901926
//owner: https://api.github.com/users/cjwon0827

import java.util.Scanner;

public class project8 {
    final static int SALARY_1 = 12000000;
    final static int SALARY_2 = 46000000;
    final static int SALARY_3 = 88000000;
    final static int SALARY_4 = 150000000;
    final static int SALARY_5 = 300000000;
    final static int SALARY_6 = 500000000;
    final static int SALARY_7 = 1000000000;

    final static int DEDUCTION_1 = 1080000;
    final static int DEDUCTION_2 = 5220000;
    final static int DEDUCTION_3 = 14900000;
    final static int DEDUCTION_4 = 19400000;
    final static int DEDUCTION_5 = 25400000;
    final static int DEDUCTION_6 = 35400000;
    final static int DEDUCTION_7 = 65400000;

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("연소득을 입력해 주세요.:");
        int salary = sc.nextInt();

        taxCal(salary);
        deductionCal(salary);

        sc.close();
    }

    private static void taxCal(int salary) {
        int taxTotal = 0;
        int result = 0;
        int result1 = (int)(SALARY_1 * 0.06);
        int result2 = (int)((SALARY_2 - SALARY_1) * 0.15);
        int result3 = (int)((SALARY_3 - SALARY_2) * 0.24);
        int result4 = (int)((SALARY_4 - SALARY_3) * 0.35);
        int result5 = (int)((SALARY_5 - SALARY_4) * 0.38);
        int result6 = (int)((SALARY_6 - SALARY_5) * 0.40);
        int result7 = (int)((SALARY_7 - SALARY_6) * 0.42);

        if(salary <= SALARY_1){
            result = (int)(salary * 0.06);
            System.out.println(salary + " * " + "6% =\t" + result + "\n");
            taxTotal += result;
        } else if(salary > SALARY_1 && salary <= SALARY_2){
            result = (int)((salary - SALARY_1) * 0.15);
            System.out.println(SALARY_1 + " * " + "6% = \t" + result1);
            System.out.println((salary - SALARY_1) + " * " + "15% = \t" + result + "\n");
            taxTotal += result + result1;
        } else if(salary > SALARY_2 && salary <= SALARY_3){
            result = (int)((salary - SALARY_2) * 0.24);
            System.out.println(SALARY_1 + " * " + "6% = \t" + result1);
            System.out.println((SALARY_2 - SALARY_1) + " * " + "15% = \t" + result2);
            System.out.println((salary - SALARY_2) + " * " + "24% = \t" + result + "\n");
            taxTotal += result + result1 + result2;
        } else if(salary > SALARY_3 && salary <= SALARY_4){
            result = (int)((salary - SALARY_3) * 0.35);
            System.out.println(SALARY_1 + " * " + "6% = \t" + result1);
            System.out.println((SALARY_2 - SALARY_1) + " * " + "15% = \t" + result2);
            System.out.println((SALARY_3 - SALARY_2) + " * " + "24% = \t" + result3);
            System.out.println((salary - SALARY_3) + " * " + "35% = \t" + result + "\n");
            taxTotal += result + result1 + result2 + result3;
        } else if(salary > SALARY_4 && salary <= SALARY_5){
            result = (int)((salary - SALARY_4) * 0.38);
            System.out.println(SALARY_1 + " * " + "6% = \t" + result1);
            System.out.println((SALARY_2 - SALARY_1) + " * " + "15% = \t" + result2);
            System.out.println((SALARY_3 - SALARY_2) + " * " + "24% = \t" + result3);
            System.out.println((SALARY_4 - SALARY_3) + " * " + "35% = \t" + result4);
            System.out.println((salary - SALARY_4) + " * " + "38% = \t" + result + "\n");
            taxTotal += result + result1 + result2 + result3 + result4;
        } else if(salary > SALARY_5 && salary <= SALARY_6){
            result = (int)((salary - SALARY_5) * 0.40);
            System.out.println(SALARY_1 + " * " + "6% = \t" + result1);
            System.out.println((SALARY_2 - SALARY_1) + " * " + "15% = \t" + result2);
            System.out.println((SALARY_3 - SALARY_2) + " * " + "24% = \t" + result3);
            System.out.println((SALARY_4 - SALARY_3) + " * " + "35% = \t" + result4);
            System.out.println((SALARY_5 - SALARY_4) + " * " + "38% = \t" + result5);
            System.out.println((salary - SALARY_5) + " * " + "40% = \t" + result + "\n");
            taxTotal += result + result1 + result2 + result3 + result4 + result5;
        } else if(salary > SALARY_6 && salary <= SALARY_7) {
            result = (int)((salary - SALARY_6) * 0.42);
            System.out.println(SALARY_1 + " * " + "6% = \t" + result1);
            System.out.println((SALARY_2 - SALARY_1) + " * " + "15% = \t" + result2);
            System.out.println((SALARY_3 - SALARY_2) + " * " + "24% = \t" + result3);
            System.out.println((SALARY_4 - SALARY_3) + " * " + "35% = \t" + result4);
            System.out.println((SALARY_5 - SALARY_4) + " * " + "38% = \t" + result5);
            System.out.println((SALARY_6 - SALARY_5) + " * " + "40% = \t" + result6);
            System.out.println((salary - SALARY_6) + " * " + "42% = \t" + result + "\n");
            taxTotal += result + result1 + result2 + result3 + result4 + result5 + result6;
        } else {
            result = (int)((salary - SALARY_7) * 0.45);
            System.out.println(SALARY_1 + " * " + "6% = \t" + result1);
            System.out.println((SALARY_2 - SALARY_1) + " * " + "15% = \t" + result2);
            System.out.println((SALARY_3 - SALARY_2) + " * " + "24% = \t" + result3);
            System.out.println((SALARY_4 - SALARY_3) + " * " + "35% = \t" + result4);
            System.out.println((SALARY_5 - SALARY_4) + " * " + "38% = \t" + result5);
            System.out.println((SALARY_6 - SALARY_5) + " * " + "40% = \t" + result6);
            System.out.println((SALARY_7 - SALARY_6) + " * " + "42% = \t" + result7);
            System.out.println((salary - SALARY_7) + " * " + "45% = \t" + result + "\n");
            taxTotal += result + result1 + result2 + result3 + result4 + result5 + result6 + result7;
        }
        System.out.println("[세율에 의한 세금]:\t" + taxTotal);
    }

    private static void deductionCal(int salary) {
        int result = 0;
        if(salary <= SALARY_1){
            result = (int)(salary * 0.06);
        } else if(salary > SALARY_1 && salary <= SALARY_2){
            result = (int)(salary * 0.15) - DEDUCTION_1;
        } else if(salary > SALARY_2 && salary <= SALARY_3){
            result = (int)(salary * 0.24) - DEDUCTION_2;
        } else if(salary > SALARY_3 && salary <= SALARY_4){
            result = (int)(salary * 0.35) - DEDUCTION_3;
        } else if(salary > SALARY_4 && salary <= SALARY_5){
            result = (int)(salary * 0.38) - DEDUCTION_4;
        } else if(salary > SALARY_5 && salary <= SALARY_6){
            result = (int)(salary * 0.40) - DEDUCTION_5;
        } else if(salary > SALARY_6 && salary <= SALARY_7) {
            result = (int)(salary * 0.42) - DEDUCTION_6;
        } else {
            result = (int)(salary * 0.45) - DEDUCTION_7;
        }
        System.out.println("[누진공제 계산에 의한 세금] : " + result);
    }
}