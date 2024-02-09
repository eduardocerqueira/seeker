//date: 2024-02-09T17:01:47Z
//url: https://api.github.com/gists/8db930cc3b9377e1dae973568e457f48
//owner: https://api.github.com/users/LEEYOOUGU

package Assignment1;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
public class Assignment2 {
    public static void main(String[] args) throws IOException {
        System.out.println("내 좌표 입력");
        BufferedReader br= new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st = "**********"

        int[] my_Coordinate = new int[2];
        int[][] input_Coordinate = new int[10][2];
        double distance = Integer.MAX_VALUE;
        int x =0;
        int y =0;
        my_Coordinate[0] = "**********"
        my_Coordinate[1] = "**********"

        for(int i =0; i<=9;i++){
            System.out.println((i+1)+"번째 임의의 좌표를 입력해주세요: ");
            st = "**********"
            input_Coordinate[i][0] = "**********"
            input_Coordinate[i][1] = "**********"
            double temp_distance = Math.sqrt(Math.pow((my_Coordinate[0]-input_Coordinate[i][0]),2)+ Math.pow((my_Coordinate[1]-input_Coordinate[i][1]),2));

            //5번쨰 소수점까지 반영
            temp_distance = Math.floor(temp_distance * 10000.0) / 10000.0;
            if(temp_distance <=distance){
                distance = temp_distance;
                x = input_Coordinate[i][0];
                y = input_Coordinate[i][1];
            }
            for(int j = i-1; j>=0;j--){
                if((input_Coordinate[i][0] == input_Coordinate[j][0]) && (input_Coordinate[i][1] == input_Coordinate[j][1])){
                    System.out.println("다시 입력해주세요");
                    i--;
                    break;
                }
                else{
                    continue;
                }
            }
        }
        System.out.printf("나와 가장 가까운 좌표값은 (%d,%d) 입니다",x,y);
    }
}
               }
            }
        }
        System.out.printf("나와 가장 가까운 좌표값은 (%d,%d) 입니다",x,y);
    }
}
