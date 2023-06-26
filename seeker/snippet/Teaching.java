//date: 2023-06-26T16:59:56Z
//url: https://api.github.com/gists/35c38af9fbe40c8dfc27864005018c7e
//owner: https://api.github.com/users/GR-SURYANARAYANA

package Lab.Employee;

public class Teaching extends Staff{
    private final String domain,publication;

    public Teaching() {
        super();
        System.out.println("Enter the domain");
        this.domain = in.next();
        System.out.println("Enter the publication");
        this.publication = in.next();
    }

    @Override
    public String toString() {
        return "\n----------Teaching Staff Details are---------"
                +super.toString()
                +"Domain of the staff is"+domain
                +"\nPublication of the staff is "+publication;

    }

}
