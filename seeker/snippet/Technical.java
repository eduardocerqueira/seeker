//date: 2023-06-26T16:59:56Z
//url: https://api.github.com/gists/35c38af9fbe40c8dfc27864005018c7e
//owner: https://api.github.com/users/GR-SURYANARAYANA

package Lab.Employee;

public class Technical extends Staff{
    private final String skills;

    public Technical() {
        super();
        System.out.println("Enter the skills of the Staff");
        this.skills = in.next();
    }
    @Override
    public String toString() {
        return "\n----------Technical Staff Details---------\n"
                +super.toString()+
                "Skills of the Staff "+skills;
    }
}
