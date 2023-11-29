//date: 2023-11-29T16:46:31Z
//url: https://api.github.com/gists/92b9b9b3ca9c30badc454af2acd16859
//owner: https://api.github.com/users/theaman05

import java.util.*;
import java.io.*;


class FileHandling{
    // this method does not handle the exception but the caller need to handle it
    // if a method can throw a checked exception, it must indicate it in
    // the exception list using 'throws'
    static String getFileContent(String filePath) throws IOException{
        BufferedReader file = new BufferedReader(new FileReader(filePath));

        String content = "";
        String line;

        while((line = file.readLine()) != null){
            content += line;
            content += "\n";
        }

        return content;
    }
}


public class ThrowsExample{

    public static void main(String[] args){

        try{
            System.out.println(FileHandling.getFileContent("test.txt"));
        }
        catch(IOException e){
            System.out.println(e.getMessage());
        }

    }
}