//date: 2025-02-13T17:03:00Z
//url: https://api.github.com/gists/d7682719299b13b02a124a11a04e5413
//owner: https://api.github.com/users/techy-ankur

package designPatterns;

/**
 * @author ankur on 13/02/25.
 */
public class StudentBuilder {
    private int id;
    private int age;
    private String name;
    private String email;
    private String password;
    private int gradYear;
    private int attendance;
    private String batchCode;

    public int getId() {
        return id;
    }

    public StudentBuilder setId(int id) {
        this.id = id;
        return this;
    }

    public int getAge() {
        return age;
    }

    public StudentBuilder setAge(int age) {
        this.age = age;
        return this;
    }

    public String getName() {
        return name;
    }

    public StudentBuilder setName(String name) {
        this.name = name;
        return this;
    }

    public String getEmail() {
        return email;
    }

    public StudentBuilder setEmail(String email) {
        this.email = email;
        return this;
    }

    public String getPassword() {
        return password;
    }

    public StudentBuilder setPassword(String password) {
        this.password = "**********"
        return this;
    }

    public int getGradYear() {
        return gradYear;
    }

    public StudentBuilder setGradYear(int gradYear) {
        this.gradYear = gradYear;
        return this;
    }

    public int getAttendance() {
        return attendance;
    }

    public StudentBuilder setAttendance(int attendance) {
        this.attendance = attendance;
        return this;
    }

    public String getBatchCode() {
        return batchCode;
    }

    public StudentBuilder setBatchCode(String batchCode) {
        this.batchCode = batchCode;
        return this;
    }

    // This method will build student object now.
    public Student build() {
        return new Student(this);
    }
}
