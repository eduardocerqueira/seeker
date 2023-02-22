//date: 2023-02-22T17:11:48Z
//url: https://api.github.com/gists/b069822bb4fa35d831f79da4b4c3b5f2
//owner: https://api.github.com/users/tahmidh

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

class ProblemTwo {

    public static void main(String[] arg) {


    }

    public static int unique(int[] list) {


        int i = 0;
        int result = 1;
        Map<Integer, Integer> numberlist = new HashMap<Integer,Integer>();

        for (i = 0; i < list.length; i++) {

            if(numberlist.containsKey(list[i])){
                int value = numberlist.get(list[i]) + 1;
                numberlist.put(list[i], value);
            }else{
                numberlist.put(list[i], 1);
            }
        }



        return numberlist.entrySet().stream().filter(entry -> result == entry.getValue()).map(Map.Entry::getKey);
    }
}



