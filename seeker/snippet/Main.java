//date: 2023-06-26T16:59:56Z
//url: https://api.github.com/gists/35c38af9fbe40c8dfc27864005018c7e
//owner: https://api.github.com/users/GR-SURYANARAYANA

package Lab.Employee;

public class Main {
    public static void main(String[] args) {
        System.out.println("Enter the Teaching Staff Detail");
        Teaching teaching=new Teaching();
        System.out.println("Enter the Technical Staff Detail");
        Technical technical=new Technical();
        System.out.println("Enter the Contract Staff Detail");
        Contract contract=new Contract();
        System.out.println(teaching);
        System.out.println(technical);
        System.out.println(contract);
    }
}
