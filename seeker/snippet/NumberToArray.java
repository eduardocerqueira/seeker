//date: 2022-02-04T17:08:53Z
//url: https://api.github.com/gists/f7b0576bd7dcbbee06990de24a59553d
//owner: https://api.github.com/users/Oywayten

package ru.job4j.array;

import java.util.Arrays;

public class NumberToArray {
    public static int[] resolve(int number) {
        String temp = Integer.toString(number);
        int[] numbers = new int[temp.length()];
        for (int i = 0; i < temp.length(); i++) {
            numbers[i] = Integer.parseInt(String.valueOf(temp.charAt(temp.length() - 1 - i)));
        }
        return  numbers;
    }
}