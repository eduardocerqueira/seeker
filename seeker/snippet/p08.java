//date: 2024-03-14T16:59:49Z
//url: https://api.github.com/gists/d8d71acc4160363329a6ec344d70355b
//owner: https://api.github.com/users/jjm159

package org.example.assignment_03_18;

/* 정재명
 *
 */

import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

public class p08 {
    public static void main(String[] args) {
        Tax tax = new Tax();
        tax.inputAnnualIncome();
        tax.calculate();
        tax.printCalculateProcess();
        tax.printResult();
    }
}

class TaxBracket {
    int min;
    int max;
    long taxPercentage;
    int progressiveDeductionAmount;

    TaxBracket(int min, int max, long taxPercentage, int progressiveDeductionAmount) {
        this.min = min;
        this.max = max;
        this.taxPercentage = taxPercentage;
        this.progressiveDeductionAmount = progressiveDeductionAmount;
    }

    public boolean isInBracket(int amount) {
        return min < amount && amount <= max;
    }

    public int calculateAllTaxByProgressiveDeduction(int amount) {
        if (!isInBracket(amount)) return 0;
        return (int) (amount / 100.0 * taxPercentage) - progressiveDeductionAmount;
    }

    public int calculateTaxInBracket(int amount) {
        double result = amountInBracket(amount) / 100.0 * taxPercentage;
        return (int) result;
    }

    private int amountInBracket(int amount) {
        if (amount < min) return 0;
        int max = Math.min(amount, this.max);
        return max - min;
    }

    public String getCalculateProgressString(int amount) {
        int currentAmount = amountInBracket(amount);
        if (currentAmount == 0) return "";
        int result = calculateTaxInBracket(amount);
        return String.format(
                "%12d * %2d%% = %12d\n",
                currentAmount,
                taxPercentage,
                result
        );
    }
}

class Tax {
    
    private final Scanner sc = new Scanner(System.in);

    private int income = 0;
    private int rateResultAmount = 0;
    private int progressiveResultAmount = 0;

    private final List<TaxBracket> taxBracketList = List.of(
            new TaxBracket(0, 12_000_000, 6, 0),
            new TaxBracket(12_000_000, 46_000_000, 15, 1_080_000),
            new TaxBracket(46_000_000, 88_000_000, 24, 5_220_000),
            new TaxBracket(88_000_000, 150_000_000, 35, 14_900_000),
            new TaxBracket(150_000_000, 300_000_000, 38, 19_400_000),
            new TaxBracket(300_000_000, 500_000_000, 40, 25_400_000),
            new TaxBracket(500_000_000, 1_000_000_000, 42, 35_400_000),
            new TaxBracket(1_000_000_000, Integer.MAX_VALUE, 45, 65_400_000)
    );


    public void inputAnnualIncome() {
        System.out.println("[과세금액 계산 프로그램]");
        System.out.print("연소득을 입력해 주세요.:");
        this.income = sc.nextInt();
        sc.nextLine();
    }

    public void calculate() {
        this.progressiveResultAmount = taxBracketList.stream()
                .filter(bracket -> bracket.isInBracket(income))
                .map(bracket -> bracket.calculateAllTaxByProgressiveDeduction(income))
                .findFirst()
                .orElse(0);
        this.rateResultAmount = taxBracketList.stream()
                .map(bracket -> bracket.calculateTaxInBracket(income))
                .reduce(0, Integer::sum);
    }

    public void printCalculateProcess() {
        String result = taxBracketList.stream()
                .map(bracket -> bracket.getCalculateProgressString(income))
                .filter(string -> !string.isEmpty())
                .collect(Collectors.joining());
        System.out.println(result);
    }

    public void printResult() {
        String rateResult = String.format(
                "[세율에 의한 세금]:\t\t\t\t\t%d",
                rateResultAmount
        );
        String progressiveResult = String.format(
                "[누진공제 계산에 의한 세금]\t\t\t%d",
                progressiveResultAmount
        );
        System.out.println(rateResult);
        System.out.println(progressiveResult);
    }
}
