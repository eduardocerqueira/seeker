//date: 2024-08-26T16:55:19Z
//url: https://api.github.com/gists/e4352ec0f33b6be397f0e8c907e79a16
//owner: https://api.github.com/users/RamshaMohammed

import java.util.TreeSet;

class TreeSetEx {
    public static void main(String[] args)
    {
        TreeSet<String> s = new TreeSet<>();
        s.add("Ramsha");
        s.add("Firdouse");
        s.add("Mohammed");
        s.add("Rayyan");
        System.out.println(s);
        
        System.out.println(s.first());
        System.out.println(s.last());
        System.out.println(s.remove("Rayyan"));
        System.out.println(s.size());
        System.out.println(s.contains("Mohammed"));
        System.out.println(s);
     
   }
}