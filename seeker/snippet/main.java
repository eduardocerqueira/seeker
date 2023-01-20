//date: 2023-01-20T16:46:49Z
//url: https://api.github.com/gists/f787fb7b6ded5f52ce6a00681b961052
//owner: https://api.github.com/users/GG1RRka

import java.io.File;
import java.util.Scanner;
public class Main {
    public static void main(String args[]) {
        String filename = "oop";
        String format = ".txt";
        String path = "C:\\Users\\Zhang\\OneDrive";
        try {
            File file = new File(path + "/" + filename + format);
            if (file.createNewFile()) {
                System.out.println("Done, file " + filename + format + " created.");
            }
            else {
                boolean created = false;
                for (int i = 2; i < 10; i++) {
                    file = new File(path + "/" + filename + "-" + i + format);
                    if (file.createNewFile()) {
                        System.out.println("Done, file " + filename + format + " already exists, so " + filename + "-" + i + format + " created.");
                        created = true;
                        break;
                    }
                }
                if (!created) {
                    System.err.println("Error, file already exists");
                }
            }
        }
        catch (Exception e) {
            System.err.println(e);
        }
    }
}