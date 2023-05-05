//date: 2023-05-05T16:56:29Z
//url: https://api.github.com/gists/4a5b1e79b77055fb20aba2b6e3cb0906
//owner: https://api.github.com/users/z0xer

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

public class Main {
    public static void main(String[] args) throws FileNotFoundException {
        NameGenerator nameGenerator = new NameGenerator();
        nameGenerator.readDataFromFile("C:\\Users\\Asus\\Desktop\\test.txt");
        String generatedName = nameGenerator.generateName();
        System.out.println(generatedName);



    }


}


class NameGenerator {

    public static Map<String, Double> bigramProbabilities = new HashMap<>();
    public static Map<Character, Double> startingCharProbabilities = new HashMap<>();
    public static List<String> possibleStartingBigrams = new ArrayList<>();
    public static Random random = new Random();
    public static Map<String, Integer> bigramCounts = new HashMap<>();

    public static void main(String[] args) {
        readDataFromFile("data.txt");

        String generatedName = generateName();

        System.out.println(generatedName);
    }

    public static void readDataFromFile(String fileName) {
        try {
            Scanner scanner = new Scanner(new File(fileName));
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().toLowerCase();
                for (int i = 0; i < line.length() - 1; i++) {
                    String bigram = line.substring(i, i + 2);
                    bigramProbabilities.put(bigram, bigramProbabilities.getOrDefault(bigram, 0.0) + 1.0);
                    if (i == 0) {
                        char startingChar = line.charAt(0);
                        startingCharProbabilities.put(startingChar, startingCharProbabilities.getOrDefault(startingChar, 0.0) + 1.0);
                        possibleStartingBigrams.add(bigram);
                    }
                }
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        double totalCount = bigramProbabilities.values().stream().mapToDouble(Double::doubleValue).sum();
        for (Map.Entry<String, Double> entry : bigramProbabilities.entrySet()) {
            double probability = entry.getValue() / totalCount;
            bigramProbabilities.put(entry.getKey(), probability);
        }

        totalCount = startingCharProbabilities.values().stream().mapToDouble(Double::doubleValue).sum();
        for (Map.Entry<Character, Double> entry : startingCharProbabilities.entrySet()) {
            double probability = entry.getValue() / totalCount;
            startingCharProbabilities.put(entry.getKey(), probability);
        }
    }

    public static String generateName() {
       
        String startingBigram = possibleStartingBigrams.get(random.nextInt(possibleStartingBigrams.size()));
        StringBuilder nameBuilder = new StringBuilder(startingBigram);

        
        while (true) {
            String previousBigram = nameBuilder.substring(nameBuilder.length() - 2);
            char nextChar = getNextChar(previousBigram);
            if (nextChar == '\0') {
                break;
            }
            nameBuilder.append(nextChar);
        }

        return nameBuilder.toString();
    }

    private static char getNextChar(String previousBigram) {
    
        List<String> possibleNextBigrams = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : bigramCounts.entrySet()) {
            if (entry.getKey().startsWith(String.valueOf(previousBigram.charAt(1)))) {
                possibleNextBigrams.add(entry.getKey());
            }
        }

      
        if (possibleNextBigrams.isEmpty()) {
            return '\0';
        }

       
        double randomValue = random.nextDouble();
        double cumulativeProbability = 0.0;
        for (String bigram : possibleNextBigrams) {
            cumulativeProbability += bigramCounts.get(bigram);
            if (randomValue <= cumulativeProbability) {
                return bigram.charAt(1);
            }
        }

      
        String lastBigram = possibleNextBigrams.get(possibleNextBigrams.size() - 1);
        return lastBigram.charAt(1);
    }
}

