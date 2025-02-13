//date: 2025-02-13T16:48:41Z
//url: https://api.github.com/gists/0c213407885a56b4a96e61c481c38067
//owner: https://api.github.com/users/techy-ankur

public class BuilderPattern {
    public static void main(String[] args) {
        StudentBuilder sb = new StudentBuilder();
        sb.setAge(10);
        sb.setEmail("ankur@gmail.com");
        sb.setId(1);

        Student s = new Student(sb);
    }
}