//date: 2023-06-26T17:01:11Z
//url: https://api.github.com/gists/67ab51f49af2c5c77db688cff9fcffe8
//owner: https://api.github.com/users/GR-SURYANARAYANA

package Lab.Employee;

import java.util.Scanner;
import java.util.StringTokenizer;

public class Customer {
    private final String string;
    private String newString;

    public Customer(String string) {
        this.string=string;
        StringTokenizer tokenizer = "**********"
        newString = "**********"
                tokenizer.nextElement() + "," +
                tokenizer.nextElement();
    }

    @Override
    public String toString() {
        return newString;
    }
    public static void main(String[] args) {
        Scanner in=new Scanner(System.in);
        System.out.println("Enter the detail of the customer i.e <Name,dd/mm/yyyy>");
        Customer customer =new Customer(in.next());
        System.out.println(customer);
    }
}
tem.out.println(customer);
    }
}
