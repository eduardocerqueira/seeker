//date: 2022-02-23T17:06:06Z
//url: https://api.github.com/gists/61cb52331d785af72f2ba51fe773cb64
//owner: https://api.github.com/users/wsido

package com.imooc;
import java.util.Scanner;

public class test2 {
    public static void main(String[] args) {
        //班级数量
        //班级总成绩
        //班级平均分
        int sum = 0 ;
        int classnum = 3;
        int stunum = 4;
        double avg = 0;
        for( int i = 1;i <= classnum;i++){
            sum = 0;
            System.out.println("***请输入第"+i+"个班级的成绩***");
            for(int j = 1;j <= stunum;j++){
                System.out.print("请输入第"+j+"个学员的成绩:");
                int score = new Scanner(System.in).nextInt();
                sum=sum+score;
            }
            avg = sum / stunum;
            System.out.println("第"+i+"班级的平均分为:"+avg);
        }
    }
}
