//date: 2023-06-22T16:46:39Z
//url: https://api.github.com/gists/1dd1b08f5b94bd0e6b716665d5877b27
//owner: https://api.github.com/users/iamdreamerofficial

import java.util.Scanner;

Scanner scanner = new Scanner(System.in);
System.out.println("Enter your name:");
String name = scanner.nextLine();
System.out.println("Hello, " + name + "!");

// File I/O
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

try {
    File file = new File("output.txt");
    FileWriter writer = new FileWriter(file);
    writer.write("Hello, World!");
    writer.close();
} catch (IOException e) {
    e.printStackTrace();
}