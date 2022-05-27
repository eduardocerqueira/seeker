//date: 2022-05-27T16:55:39Z
//url: https://api.github.com/gists/5ac910f5fb9cd1fada9e4841e6f7686d
//owner: https://api.github.com/users/citronov

public class task5 {
    public static void main(String[] args) {

        int x = 91;

        if (x >= 0 && x <= 19) {
            System.out.println("F");
        } else if (x >= 20 && x <= 39) {
            System.out.println("E");
        } else if (x >= 40 && x <= 59) {
            System.out.println("D");
        } else if (x >= 60 && x <= 74) {
            System.out.println("C");
        } else if (x >= 75 && x <= 89) {
            System.out.println("B");
        } else if (x >= 90 && x <= 100) {
            System.out.println("A");
        }
    }
}
