//date: 2023-04-11T16:58:06Z
//url: https://api.github.com/gists/fb44945e1128bf2a6ffbf614a2b2b032
//owner: https://api.github.com/users/YesunPark

// 23.4.12 4월(6주차) Mission1_깜짝과제1 - html파일 작성해보기

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        // 파일 저장
        try {
            File file = new File("property.html");
            BufferedWriter writer = new BufferedWriter(new FileWriter(file));
            writer.write("<head>");
            writer.write("<meta charset=\"UTF-8\" />");
            writer.write("<style>" +
                    "table { border-collapse: collapse; width: 100%; }" +
                    "th, td { border:solid 1px #000;}" +
                    "</style>");
            writer.write("</head>");

            writer.write("<body>");

            writer.write("<h1>자바 환경정보</h1>");
            writer.write("<table>" +
                    "<th>키</th>" +
                    "<th></th>");
            // 자바 시스템 속성값 출력
            for (Object k : System.getProperties().keySet()) {
                String key = k.toString();
                String value = System.getProperty(key);

                writer.write("<tr>" +
                        "<td>" + key + "</td>"
                        + "<td>" + value + "</td>"
                        + "</tr>");
            }
            writer.write("</table>");
            
            writer.write("</body>");
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}