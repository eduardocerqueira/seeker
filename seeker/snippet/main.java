//date: 2021-10-27T17:17:19Z
//url: https://api.github.com/gists/d104ca68d972723fb8aa3eb203855561
//owner: https://api.github.com/users/YouRageQuitForMe

package me.yourage;

import me.yourage.ReadText;

import java.io.FileNotFoundException;
import java.util.*;

public class Main {

    public static void main(String[] args) throws FileNotFoundException, InterruptedException {

        ArrayList<String> words = new ReadText().ReadFile();
        Scanner scanWord = new Scanner(System.in);
        String userWord;
        System.out.println("Enter a word");
        userWord = scanWord.nextLine();

        Set<String> set = new TreeSet<String>();
        String key = userWord;

        //S1 = c*S0 ∪ {c} ∪ S0
        for(int z = 0; z < key.length();z++) {
            Set<String> temp = new HashSet<String>();
            char c = key.charAt(z);

            for(String str: set)
                temp.add(str + c); // ∪ c*S0
            set.add(c+"");         // ∪ {c}
            set.addAll(temp);      // ∪ S0
        }

        words.forEach((word) -> {
            set.forEach((setWord) -> {
                if (word.equals(setWord))
                    System.out.println("word:" + word);
            });
        });

    }
}
