//date: 2022-01-14T17:18:48Z
//url: https://api.github.com/gists/8e3a018a6e7970c1ba159e39a5542a78
//owner: https://api.github.com/users/Khujamov

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class Test {
    public static void main(String[] args) throws Exception {
        List<String> twoInts = Files.readAllLines(
                Paths.get("/home/khumoyun/learning/codingbat-solutions-java/input1.txt"));

        List<String> nums = Files.readAllLines(
                Paths.get("/home/khumoyun/learning/codingbat-solutions-java/input2.txt"));
        String[] split = twoInts.get(0).split(",");
        int firstNum = Integer.parseInt(split[0]);
        int secondNum = Integer.parseInt(split[1]);

        StringBuilder builder = new StringBuilder();
        nums.forEach(s -> {
            String[] numsByLine = s.split(" ");
            List<String> numsAsList = Arrays.asList(numsByLine);
            numsAsList.stream()
                    .mapToInt(Integer::parseInt)
                    .forEach(value -> {
                        if (value % firstNum == 0 && value % secondNum == 0) {
                            builder.append("HotDog");
                        } else if (value % firstNum == 0) {
                            builder.append("Hot");
                        } else if (value % secondNum == 0) {
                            builder.append("Dog");
                        } else builder.append(value);
                        builder.append(" ");
                    });
        });
        System.out.println(builder);
    }
}
