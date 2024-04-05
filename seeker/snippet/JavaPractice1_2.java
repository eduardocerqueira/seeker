//date: 2024-04-05T16:47:31Z
//url: https://api.github.com/gists/67b4f13826b4e37f3272ca5b3f17f505
//owner: https://api.github.com/users/KimSoyoung-TOPGUN

//김소영

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class practice1_2 {
    public static String getLoadText(String filePath) { //정적 함수로 맞춰줘야 함
        StringBuilder sb = new StringBuilder();

        try {
            Path path = Paths.get(filePath);
            List<String> lines = Files.readAllLines(path);
            for (int i = 0; i < lines.size(); i++) {
                if (i > 0) {
                    sb.append("\n");
                }
                sb.append(lines.get(i));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return sb.toString();
    }

    public static void main(String[] args) { //정적 함수
        String filePath = "/Users/ksy/Downloads/practice1_2.txt";

        String line = getLoadText(filePath); //getLoadText 함수를 통해서 얻은 문자열을 저장
        int lineLength = line.length(); //전체 문자열 길이

        //각각의 알파벳 개수
        int countA = 0, countB = 0, countC = 0, countD = 0, countE = 0, countF = 0, countG = 0, countH = 0, countI = 0, countJ = 0, countK = 0, countL = 0, countM = 0, countN = 0, countO = 0, countP = 0, countQ = 0, countR = 0, countS = 0, countT = 0, countU = 0, countV = 0, countW = 0, countX = 0, countY = 0, countZ = 0;

        //전체 알파벳 개수
        int totalCount = 0;

        for (int i = 0; i < lineLength; i++) {
            switch (Character.toUpperCase(line.charAt(i))) {
                case 'A' : countA++; totalCount++; break;
                case 'B' : countB++; totalCount++; break;
                case 'C' : countC++; totalCount++; break;
                case 'D' : countD++; totalCount++; break;
                case 'E' : countE++; totalCount++; break;
                case 'F' : countF++; totalCount++; break;
                case 'G' : countG++; totalCount++; break;
                case 'H' : countH++; totalCount++; break;
                case 'I' : countI++; totalCount++; break;
                case 'J' : countJ++; totalCount++; break;
                case 'K' : countK++; totalCount++; break;
                case 'L' : countL++; totalCount++; break;
                case 'M' : countM++; totalCount++; break;
                case 'N' : countN++; totalCount++; break;
                case 'O' : countO++; totalCount++; break;
                case 'P' : countP++; totalCount++; break;
                case 'Q' : countQ++; totalCount++; break;
                case 'R' : countR++; totalCount++; break;
                case 'S' : countS++; totalCount++; break;
                case 'T' : countT++; totalCount++; break;
                case 'U' : countU++; totalCount++; break;
                case 'V' : countV++; totalCount++; break;
                case 'W' : countW++; totalCount++; break;
                case 'X' : countX++; totalCount++; break;
                case 'Y' : countY++; totalCount++; break;
                case 'Z' : countZ++; totalCount++; break;
            }
        }

        System.out.println("A =    " + String.format("%3d개,    ", countA) + String.format("%.2f", (double) countA / (double) totalCount * 100.0) + "%"); //%[최소길이]s: 최소 길이만큼 "숫자" 출력, 왼쪽에서부터 공백으로 채움
        System.out.println("B =    " + String.format("%3d개,    ", countB) + String.format("%.2f", (double) countB / (double) totalCount * 100.0) + "%");
        System.out.println("C =    " + String.format("%3d개,    ", countC) + String.format("%.2f", (double) countC / (double) totalCount * 100.0) + "%");
        System.out.println("D =    " + String.format("%3d개,    ", countD) + String.format("%.2f", (double) countD / (double) totalCount * 100.0) + "%");
        System.out.println("E =    " + String.format("%3d개,    ", countE) + String.format("%.2f", (double) countE / (double) totalCount * 100.0) + "%");
        System.out.println("F =    " + String.format("%3d개,    ", countF) + String.format("%.2f", (double) countF / (double) totalCount * 100.0) + "%");
        System.out.println("G =    " + String.format("%3d개,    ", countG) + String.format("%.2f", (double) countG / (double) totalCount * 100.0) + "%");
        System.out.println("H =    " + String.format("%3d개,    ", countH) + String.format("%.2f", (double) countH / (double) totalCount * 100.0) + "%");
        System.out.println("I =    " + String.format("%3d개,    ", countI) + String.format("%.2f", (double) countI / (double) totalCount * 100.0) + "%");
        System.out.println("J =    " + String.format("%3d개,    ", countJ) + String.format("%.2f", (double) countJ / (double) totalCount * 100.0) + "%");
        System.out.println("K =    " + String.format("%3d개,    ", countK) + String.format("%.2f", (double) countK / (double) totalCount * 100.0) + "%");
        System.out.println("L =    " + String.format("%3d개,    ", countL) + String.format("%.2f", (double) countL / (double) totalCount * 100.0) + "%");
        System.out.println("M =    " + String.format("%3d개,    ", countM) + String.format("%.2f", (double) countM / (double) totalCount * 100.0) + "%");
        System.out.println("N =    " + String.format("%3d개,    ", countN) + String.format("%.2f", (double) countN / (double) totalCount * 100.0) + "%");
        System.out.println("O =    " + String.format("%3d개,    ", countO) + String.format("%.2f", (double) countO / (double) totalCount * 100.0) + "%");
        System.out.println("P =    " + String.format("%3d개,    ", countP) + String.format("%.2f", (double) countP / (double) totalCount * 100.0) + "%");
        System.out.println("Q =    " + String.format("%3d개,    ", countQ) + String.format("%.2f", (double) countQ / (double) totalCount * 100.0) + "%");
        System.out.println("R =    " + String.format("%3d개,    ", countR) + String.format("%.2f", (double) countR / (double) totalCount * 100.0) + "%");
        System.out.println("S =    " + String.format("%3d개,    ", countS) + String.format("%.2f", (double) countS / (double) totalCount * 100.0) + "%");
        System.out.println("T =    " + String.format("%3d개,    ", countT) + String.format("%.2f", (double) countT / (double) totalCount * 100.0) + "%");
        System.out.println("U =    " + String.format("%3d개,    ", countU) + String.format("%.2f", (double) countU / (double) totalCount * 100.0) + "%");
        System.out.println("V =    " + String.format("%3d개,    ", countV) + String.format("%.2f", (double) countV / (double) totalCount * 100.0) + "%");
        System.out.println("W =    " + String.format("%3d개,    ", countW) + String.format("%.2f", (double) countW / (double) totalCount * 100.0) + "%");
        System.out.println("X =    " + String.format("%3d개,    ", countX) + String.format("%.2f", (double) countX / (double) totalCount * 100.0) + "%");
        System.out.println("Y =    " + String.format("%3d개,    ", countY) + String.format("%.2f", (double) countY / (double) totalCount * 100.0) + "%");
        System.out.println("Z =    " + String.format("%3d개,    ", countZ) + String.format("%.2f", (double) countZ / (double) totalCount * 100.0) + "%");
    }
}