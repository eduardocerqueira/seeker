//date: 2026-03-12T17:44:18Z
//url: https://api.github.com/gists/2ea318e94ef0887ee3726bef5e84baf8
//owner: https://api.github.com/users/aryansoni25

package org.studyeasy;
import java.util.*;

public class hello {
    public static void main(String[] args) {
        Set<String>  s1=new HashSet<>();
        s1.add("A");
        s1.add("B");
        s1.add("C");
        System.out.println(s1);
        System.out.println(s1.contains("D"));
        s1.remove("C");
        System.out.println(s1);
        System.out.println(s1.size());
    }
}
