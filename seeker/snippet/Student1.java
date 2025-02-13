//date: 2025-02-13T16:49:45Z
//url: https://api.github.com/gists/42ed5b4bc8f9262a5057c46c69da5164
//owner: https://api.github.com/users/techy-ankur

package designPatterns;

/**
 * @author ankur on 13/02/25.
 */
public class Student {
    private int id;
    private int age;
    private String name;
    private String email;
    private String password;
    private int gradYear;
    private int attendance;
    private String batchCode;

    public Student(StudentBuilder sb) {
        this.id = sb.getId();
        this.age = sb.getAge();
        this.name = sb.getName();
        this.email = sb.getEmail();
        this.password = "**********"
        this.gradYear = sb.getGradYear();
        this.attendance = sb.getAttendance();
        this.batchCode = sb.getBatchCode();
    }
}
 }
}
