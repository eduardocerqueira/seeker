//date: 2022-06-30T16:53:32Z
//url: https://api.github.com/gists/c2cd8c4dee4fb7ddb38a88e573cba2c1
//owner: https://api.github.com/users/ansis-m

package converter;

import java.math.BigInteger;
import java.util.Scanner;

public class Main {

    static final char[] xdec = new char[]{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'b', 'c', 'd', 'e', 'f'};

    public static String fromDecimal(BigInteger base, BigInteger decimal) { //convert integer part from decimal to given base

        if(decimal.equals(BigInteger.ZERO))
            return "0";

        StringBuilder result = new StringBuilder();

        while(decimal.compareTo(BigInteger.ZERO) != 0) {
            result.insert(0, xdec[decimal.mod(base).intValue()]);
            decimal = decimal.divide(base);
        }

        return result.toString();
    }

    public static String fromDecimalFraction(Double base, Double fraction) { //convert fraction part from decimal to given base

        StringBuilder result = new StringBuilder(".");

        for(int i = 0; i < 5; i++) {
            fraction *= base;
            result.append(xdec[fraction.intValue()]);
            fraction = fraction - fraction.intValue();
        }
        return result.toString();
    }

    public static BigInteger toDecimal(BigInteger base, String number) { //convert integer part to decimal base

        char[] digits =  number.toCharArray();

        BigInteger result = BigInteger.ZERO;
        for (char digit : digits) {
            result = result.multiply(base);
            for (int j = 0; j < base.longValue(); j++) {
                if (xdec[j] == digit) {
                    result = result.add(BigInteger.valueOf(j));
                    break;
                }
            }
        }
        return result;
    }

    public static double toDecimalFraction(double base, String number) { //convert fraction in given base to decimal fraction

        char[] digits =  number.toCharArray();
        double denominator = 1.000000D;
        double numerator = 0.000000D;
        for (char digit : digits) {
            numerator = numerator * base;
            denominator = denominator * base;
            for (int j = 0; j < base; j++) {
                if (xdec[j] == digit) {
                    numerator = numerator + j;
                    break;
                }
            }
        }
        return numerator / denominator;
    }


    static void convert(String answer, Scanner scanner) {

        String[] bases = answer.split(" ");
        while(true){
            System.out.println("Enter number in base " + bases[0] + " to convert to base " + bases[1] + " (To go back type /back)");
            String number = scanner.nextLine();
            if (number.equals("/back"))
                return;
            String[] splitNumber = number.split("\\."); //split the fractional number in integer and fractional part
            BigInteger decimal = toDecimal(new BigInteger(bases[0]), splitNumber[0]); //convert the integer part to decimal
            String baseFraction = "";
            if(splitNumber.length == 2) { //if fractional part is present it is converted to decimal and then to base
                double decimalFraction = toDecimalFraction(Double.parseDouble(bases[0]), splitNumber[1]);
                baseFraction = fromDecimalFraction(Double.valueOf(bases[1]), decimalFraction);
            }
            String result = fromDecimal(new BigInteger(bases[1]), decimal);
            System.out.println("Conversion result: " + result + baseFraction);
        }
    }

    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);

        while(true){
            System.out.println("Enter two numbers in format: {source base} {target base} (To quit type /exit)");
            String bases = scanner.nextLine();
            if (bases.equals("/exit")) //exit the infinite loop
                break;
            convert(bases, scanner); //moving to convert function if base numbers are provided
        }
        scanner.close();
    }
}
