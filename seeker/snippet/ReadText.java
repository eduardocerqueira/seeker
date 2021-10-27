//date: 2021-10-27T17:14:05Z
//url: https://api.github.com/gists/b0f47dbb6747c660fd2e970f116a7f8f
//owner: https://api.github.com/users/YouRageQuitForMe

package me.yourage;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class ReadText {

    public ArrayList<String> ReadFile() throws FileNotFoundException {

        ArrayList<String> listOfWords = new ArrayList<>();

            File dictionary = new File("/Users/user/Documents/java/halloweenchallengeJava/src/me/yourage/dictionary.txt");
            Scanner reader = new Scanner(dictionary);
            while (reader.hasNextLine()) {
                String word = reader.nextLine();
                listOfWords.add(word);
            }
    return listOfWords;
    }
}
