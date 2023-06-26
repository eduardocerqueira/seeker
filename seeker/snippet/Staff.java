//date: 2023-06-26T16:59:56Z
//url: https://api.github.com/gists/35c38af9fbe40c8dfc27864005018c7e
//owner: https://api.github.com/users/GR-SURYANARAYANA

package Lab.Employee;

import java.util.Scanner;

public class Staff {
    Scanner in=new Scanner(System.in);
    private final String id,name,phoneNO,salary;
    protected Staff() {
        System.out.println("Enter the id");
        this.id = in.next();
        System.out.println("Enter the name");
        this.name = in.next();
        System.out.println("Enter the phone Number ");
        this.phoneNO = in.next();
        System.out.println("Enter the salary ");
        this.salary = in.next();
    }

    @Override
    public String toString() {
        return "\nStaff id is "+ id + '\n' +
                "Staff name is " + name + '\n' +
                "Staff phoneNO is " + phoneNO + '\n' +
                "Staff salary is " + salary + '\n';
    }
}
