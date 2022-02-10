//date: 2022-02-10T16:43:40Z
//url: https://api.github.com/gists/ee646da16066f5e1392741b83f524e70
//owner: https://api.github.com/users/lismsdh2

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

public class Q3 {
    public static void main(String[] args) {
        try{
            File file = new File("Presidential speech.txt");
            FileReader filereader = new FileReader(file);
            BufferedReader bufReader = new BufferedReader(filereader);
            
            String line = "";
            int strLen = 0;
            int a = 0;
            int b = 0;
            int c = 0;
            int d = 0;
            int e = 0;
            int f = 0;
            int g = 0;
            int h = 0;
            int i = 0;
            int j = 0;
            int k = 0;
            int l = 0;
            int m = 0;
            int n = 0;
            int o = 0;
            int p = 0;
            int q = 0;
            int r = 0;
            int s = 0;
            int t = 0;
            int u = 0;
            int v = 0;
            int w = 0;
            int x = 0;
            int y = 0;
            int z = 0;

            while((line = bufReader.readLine()) != null){
                String match = "[^\uAC00-\uD7A30-9a-zA-Z]";
                String line2 = line.replaceAll(match, "").toLowerCase();
                strLen += line2.length();
                for(int ii = 0 ; ii < line2.length() ; ii++){
                    char c1 = line2.charAt(ii);
                    switch(c1){
                        case 'a':
                            a++;
                            break;
                        case 'b':
                            b++;
                            break;
                        case 'c':
                            c++;
                            break;
                        case 'd':
                            d++;
                            break;
                        case 'e':
                            e++;
                            break;
                        case 'f':
                            f++;
                            break;
                        case 'g':
                            g++;
                            break;
                        case 'h':
                            h++;
                            break;
                        case 'i':
                            i++;
                            break;
                        case 'j':
                            j++;
                            break;
                        case 'k':
                            k++;
                            break;
                        case 'l':
                            l++;
                            break;
                        case 'm':
                            m++;
                            break;
                        case 'n':
                            n++;
                            break;
                        case 'o':
                            o++;
                            break;
                        case 'p':
                            p++;
                            break;
                        case 'q':
                            q++;
                            break;
                        case 'r':
                            r++;
                            break;
                        case 's':
                            s++;
                            break;
                        case 't':
                            t++;
                            break;
                        case 'u':
                            u++;
                            break;
                        case 'v':
                            v++;
                            break;
                        case 'w':
                            w++;
                            break;
                        case 'x':
                            x++;
                            break;
                        case 'y':
                            y++;
                            break;
                        case 'z':
                            z++;
                            break;
                    }
                }
            }

            System.out.println("총 문자 수 : "+strLen+"개");
            System.out.println("A : " + a +"개, "+String.format("%.2f",((double)a/(double)strLen*100.0))+"%");
            System.out.println("B : " + b +"개, "+String.format("%.2f",((double)b/(double)strLen*100.0))+"%");
            System.out.println("C : " + c +"개, "+String.format("%.2f",((double)c/(double)strLen*100.0))+"%");
            System.out.println("D : " + d +"개, "+String.format("%.2f",((double)d/(double)strLen*100.0))+"%");
            System.out.println("E : " + e +"개, "+String.format("%.2f",((double)e/(double)strLen*100.0))+"%");
            System.out.println("F : " + f +"개, "+String.format("%.2f",((double)f/(double)strLen*100.0))+"%");
            System.out.println("G : " + g +"개, "+String.format("%.2f",((double)g/(double)strLen*100.0))+"%");
            System.out.println("H : " + h +"개, "+String.format("%.2f",((double)h/(double)strLen*100.0))+"%");
            System.out.println("I : " + i +"개, "+String.format("%.2f",((double)i/(double)strLen*100.0))+"%");
            System.out.println("J : " + j +"개, "+String.format("%.2f",((double)j/(double)strLen*100.0))+"%");
            System.out.println("K : " + k +"개, "+String.format("%.2f",((double)k/(double)strLen*100.0))+"%");
            System.out.println("L : " + l +"개, "+String.format("%.2f",((double)l/(double)strLen*100.0))+"%");
            System.out.println("M : " + m +"개, "+String.format("%.2f",((double)m/(double)strLen*100.0))+"%");
            System.out.println("N : " + n +"개, "+String.format("%.2f",((double)n/(double)strLen*100.0))+"%");
            System.out.println("O : " + o +"개, "+String.format("%.2f",((double)o/(double)strLen*100.0))+"%");
            System.out.println("P : " + p +"개, "+String.format("%.2f",((double)p/(double)strLen*100.0))+"%");
            System.out.println("Q : " + q +"개, "+String.format("%.2f",((double)q/(double)strLen*100.0))+"%");
            System.out.println("R : " + r +"개, "+String.format("%.2f",((double)r/(double)strLen*100.0))+"%");
            System.out.println("S : " + s +"개, "+String.format("%.2f",((double)s/(double)strLen*100.0))+"%");
            System.out.println("T : " + t +"개, "+String.format("%.2f",((double)t/(double)strLen*100.0))+"%");
            System.out.println("U : " + u +"개, "+String.format("%.2f",((double)u/(double)strLen*100.0))+"%");
            System.out.println("V : " + v +"개, "+String.format("%.2f",((double)v/(double)strLen*100.0))+"%");
            System.out.println("W : " + w +"개, "+String.format("%.2f",((double)w/(double)strLen*100.0))+"%");
            System.out.println("X : " + x +"개, "+String.format("%.2f",((double)x/(double)strLen*100.0))+"%");
            System.out.println("Y : " + y +"개, "+String.format("%.2f",((double)y/(double)strLen*100.0))+"%");
            System.out.println("Z : " + z +"개, "+String.format("%.2f",((double)z/(double)strLen*100.0))+"%");

            bufReader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
