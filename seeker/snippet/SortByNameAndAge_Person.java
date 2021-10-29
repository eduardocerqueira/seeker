//date: 2021-10-29T17:09:34Z
//url: https://api.github.com/gists/642b92716ba0ec18ac23fd4b86650489
//owner: https://api.github.com/users/TsvetoslavBorisov

package SortByNameAndAge;

public class Person {
    private String firstName;
    private String lastName;
    private int age;

    public Person(String firstName, String lastName, int age) {
        this.firstName = firstName;
        this.lastName = lastName;
        this.age = age;
    }

    public String getFirstName() {
        return firstName;
    }

    public int getAge() {
        return age;
    }

    @Override
    public String toString() {
        return String.format("%s %s is %d years old.", firstName, lastName, age);
    }
}
