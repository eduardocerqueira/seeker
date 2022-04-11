//date: 2022-04-11T17:17:34Z
//url: https://api.github.com/gists/57734e87828dd42c0fad321b1bf00427
//owner: https://api.github.com/users/xfoxcom

import java.util.*;
import java.util.regex.*;

public class Main {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int size = Integer.parseInt(scanner.nextLine());
        String line = scanner.nextLine();

        // write your code here
        Pattern pattern = Pattern.compile("\\b([a-zA-Z]" + "{" + size + "})\\b");
        Matcher matcher = pattern.matcher(line);
        if (matcher.find()) System.out.println("YES");
        else System.out.println("NO");
    }
}