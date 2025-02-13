//date: 2025-02-13T16:55:11Z
//url: https://api.github.com/gists/23d1d29ddf20adf2f456356864d91a65
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

    // new studentBuilder calling method
    public static StudentBuilder getBuilder() {
        return new StudentBuilder();
    }
}
 }
}
