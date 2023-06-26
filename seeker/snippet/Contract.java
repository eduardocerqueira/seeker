//date: 2023-06-26T16:59:56Z
//url: https://api.github.com/gists/35c38af9fbe40c8dfc27864005018c7e
//owner: https://api.github.com/users/GR-SURYANARAYANA

package Lab.Employee;

public class Contract extends Staff {
    private final int period;

    public Contract() {
        super();
        System.out.println("Enter the Contract Period");
        this.period=in.nextInt();
    }

    @Override
    public String toString() {
          return  "\n-----------------Contract staff details---------\n"
                  +super.toString()+
                  "Staff  Contract period is of " + period+ " years";
    }
}
