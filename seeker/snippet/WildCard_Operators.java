//date: 2024-08-26T16:55:19Z
//url: https://api.github.com/gists/e4352ec0f33b6be397f0e8c907e79a16
//owner: https://api.github.com/users/RamshaMohammed

import java.util.*;

class Example 
{
    public static void output(List<?> l) 
    {
        for (Object obj : l) 
        {
            String str = (String) obj;  
            System.out.println(str);
        }
    }

    public static void display(List<? extends Number> l) 
    {
        for (int i = 0; i < l.size(); i++) 
        {
            System.out.println(l.get(i));
        }
    }
}

public class WildCard_Operators
{
    public static void main(String[] args) 
    {
        List<String> str = new ArrayList<>();
        str.add("Ramsha");
        str.add("Yeshu");
        str.add("Likitha");
        str.add("Yamini");
        System.out.println("Names:");
        Example.output(str);  
        
        List<Integer> p = new ArrayList<>();
        p.add(45);
        p.add(67);
        System.out.println("Integers:");
        Example.display(p);  

    }
}