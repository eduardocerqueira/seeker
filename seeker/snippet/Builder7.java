//date: 2025-02-13T17:04:50Z
//url: https://api.github.com/gists/e473d759e4523ceec1e98e19bd1b5c3c
//owner: https://api.github.com/users/techy-ankur

package designPatterns;

/**
 * @author ankur on 13/02/25.
 */
public class BuilderPattern {
    public static void main(String[] args) {
        Student student = Student.getBuilder().setAge(10).setEmail("ankur@gmail.com").setId(1).build();
    }
}
