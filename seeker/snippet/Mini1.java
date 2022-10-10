//date: 2022-10-10T17:10:16Z
//url: https://api.github.com/gists/c1cb1108fdeeb591153ddabc1a801053
//owner: https://api.github.com/users/lusiun

public class n001 {

        public static void main(String[] args) {
            System.out.println("[구구단]");

            for (int i = 2; i <= 9; i++) {
                for (int j = 1; j <= 9; j++) {
                    System.out.print(j + " x " + i + " = " + String.format("%2d", i * j));
                    System.out.print("    ");
                }
                System.out.println();
            }
        }
    }