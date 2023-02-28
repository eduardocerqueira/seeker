//date: 2023-02-28T17:01:11Z
//url: https://api.github.com/gists/413717315caceb093fe9f5f3907fe4e8
//owner: https://api.github.com/users/neziw

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class Headers {

    public static void main(String[] args) throws IOException {
        // Path to folder that contains .java files (files in subfolders will be auto-detected)
        String folderPath = "C:\\Users\\neziw\\Documents\\app\\src\\main\\java\\xyz\\neziw\\app";
        // Path to header file (your header must starts with /* and ends with */ to make it as comment
        String headerFilePath = "C:\\Users\\neziw\\Documents\\app\\src\\main\\resources\\header.txt";
        String header = new String(Files.readAllBytes(Paths.get(headerFilePath)));
        try (Stream<Path> paths = Files.walk(Paths.get(folderPath))) {
            paths.filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .forEach(path -> {
                        try {
                            String content = new String(Files.readAllBytes(path));
                            // Write header before "package" keyword
                            String modifiedContent = content.replaceAll("^package", header + "\n\npackage");
                            FileWriter writer = new FileWriter(path.toString());
                            writer.write(modifiedContent);
                            writer.close();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    });
        }
    }
}