//date: 2023-07-18T17:05:35Z
//url: https://api.github.com/gists/1af89be6c49130999c6b81d53220b91f
//owner: https://api.github.com/users/yash8917

// HashSet store in according to the HashCode.
// Stores Unique Value.
// Support the null value.
// stores ele using hashing mechanism.
// not maintain the insertion order

import java.util.HashSet;

public class Demo5 {
    public static void main(String[] args) {
        HashSet set  = new HashSet();
        set.add(1);
        set.add(2);
        set.add(null);
        set.add(4);
        set.forEach(n -> System.out.println(n));
    }

}
