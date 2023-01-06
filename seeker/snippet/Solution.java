//date: 2023-01-06T16:47:02Z
//url: https://api.github.com/gists/9da2f325ab40cfd3011dbbc532c8e403
//owner: https://api.github.com/users/ram0973

package com.javarush.task.task19.task1920;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

/* 
Самый богатый
*/

public class Solution {
    public static void main(String[] args) {
        try (BufferedReader br = new BufferedReader(new FileReader(args[0]))) {
            br.lines()
                    .map(line -> line.split(" "))
                    .collect(Collectors.toMap(e -> e[0], e -> Double.parseDouble(e[1]), Double::sum))
                    .entrySet()
                    .stream()
                    .collect(Collectors.groupingBy(Map.Entry::getValue, TreeMap::new, 
                            Collectors.mapping(Map.Entry::getKey, Collectors.toList())))
                    .lastEntry()
                    .getValue()
                    .stream()
                    .sorted()
                    .forEach(System.out::println);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (NullPointerException ignored) {
        }
    }
}