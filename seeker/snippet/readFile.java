//date: 2024-04-26T16:51:40Z
//url: https://api.github.com/gists/9b3c588fbc003048995a0da94a157a09
//owner: https://api.github.com/users/aluoch-dev

package FileOperations;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class ReadFile {
    public static void main(String[] args) {
        try {
            File myFileObject = new File("filename.txt");
            Scanner myReader = new Scanner(myFileObject);

            while(myReader.hasNextLine()) {
                String data = myReader.nextLine();
                System.out.println(data);
            }

            myReader.close();

        } catch(FileNotFoundException e) {
            System.out.println("An error occurred!");
            e.printStackTrace();
        }

    }
}