//date: 2025-02-13T16:56:02Z
//url: https://api.github.com/gists/8b13680ada107d9e6521061aa2afbad1
//owner: https://api.github.com/users/techy-ankur

package designPatterns;

/**
 * @author ankur on 13/02/25.
 */
public class BuilderPattern {
    public static void main(String[] args) {
        // Student class is now creating builder itself.
        StudentBuilder sb = Student.getBuilder();
        sb.setAge(10);
        sb.setEmail("ankur@gmail.com");
        sb.setId(1);

        Student s = new Student(sb);
    }
}