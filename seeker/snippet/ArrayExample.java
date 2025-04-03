//date: 2025-04-03T16:48:12Z
//url: https://api.github.com/gists/3ae23382860a0b2e77d7dfe42c4794a5
//owner: https://api.github.com/users/jraghu15

import java.util.ArrayList;

public class ArrayExample {
    public static void main(String[] args) {
        ArrayList<String> list = new ArrayList<String>();
        list.add("A");
        list.add("B");
        list.add("C");
        list.add("A");
        list.add(2,"B");
        System.out.println(list.size());
        System.out.println(list);
    }
}
