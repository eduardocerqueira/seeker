//date: 2024-08-26T16:55:19Z
//url: https://api.github.com/gists/e4352ec0f33b6be397f0e8c907e79a16
//owner: https://api.github.com/users/RamshaMohammed

import java.util.TreeMap;

class TreeMapEx {
    public static void main(String[] args) {
        
        TreeMap<String, Integer> m = new TreeMap<>();
        m.put("gold", 3);
        m.put("Silver", 5);
        m.put("Diamond", 7);
        m.put("Platinum", 9);
    
        System.out.println(m);
        System.out.println("First Entry: " + m.firstEntry());
        System.out.println("Last Entry: " + m.lastEntry());
        System.out.println("Size of the TreeMap: " + m.size());

        boolean containsSilver = m.containsKey("Silver");
        System.out.println("TreeMap contains 'Silver': " + containsSilver);
        System.out.println(m.get("Diamond"));
        System.out.println(m.remove("gold"));
        System.out.println(m.replace("Diamond", 6));
        System.out.println("Final TreeMap: " + m);
    }
}