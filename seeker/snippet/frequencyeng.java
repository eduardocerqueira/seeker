//date: 2022-12-09T16:52:12Z
//url: https://api.github.com/gists/2de7668e053dba5ef272b7389b7bf5fd
//owner: https://api.github.com/users/icebreaker382

// 깜짝문제2
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
import java.nio.file.Path;
import java.nio.file.Files;
public class frequencyeng {

    public static String getLoadText(String filePath){
        StringBuilder sb = new StringBuilder();

        try{
            Path path = Paths.get(filePath);
            List<String> lines = Files.readAllLines(path);
            for(int i=0; i<lines.size(); i++){
                if(i > 0){
                    sb.append("\n");
                }
                sb.append(lines.get(i));
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return sb.toString();
    }

    public static void countch(){
        String save_sb = new String();
        save_sb = getLoadText("src/address.txt");
        double[] atoz = new double[26];
        for(int i=0; i<atoz.length; i++) atoz[i] = 0;
        double count = 0;

        // A-Z count
        for(int i=0; i<save_sb.length(); i++){
            count++;
            if((save_sb.charAt(i)>='A')&& (save_sb.charAt(i)<='Z')){
                int index = save_sb.charAt(i) - (int)'A';
                atoz[index]++;
            }
            else if((save_sb.charAt(i)>='a')&&(save_sb.charAt(i)<='z')){
                int index = save_sb.charAt(i) - (int)'a';
                atoz[index]++;
            }
        }

        int ch = 65;
        for(int i=0; i<atoz.length; i++){
            double rate = atoz[i] / count *100;
            String myword = String.format("%c =     %4.0f개,    %.02f%%",ch,atoz[i], rate);
            System.out.println(myword);
            ch++;
        }

    }
    public static void main(String[] args) {
        countch();
    }
}