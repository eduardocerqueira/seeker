//date: 2022-03-22T17:11:02Z
//url: https://api.github.com/gists/f985ed69bc87ce401eed97c24ed7a569
//owner: https://api.github.com/users/laxmi2001

package CAT2.Practice;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class FileWrite {
    public static void main(String[] args) {
        try(BufferedWriter bw = new BufferedWriter(new FileWriter("trial.txt", true))){
            String content = "This message is written for trial purposes1";
            bw.write(content);
            bw.write("\n");
            System.out.println("Write success");
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }
}
