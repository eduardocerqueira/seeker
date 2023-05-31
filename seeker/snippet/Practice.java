//date: 2023-05-31T16:51:42Z
//url: https://api.github.com/gists/d8e76368a7a50c9f2e4edc6cab1bfbb7
//owner: https://api.github.com/users/yeon-ju-k


import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;

public class JavaTest {

    // # 8. 기수 정렬
    public static void radixSort(int[] arr){

        // ① 기수 테이블 생성 (큐로 생성)
        //  ㄴ ArrayList 에 담긴 큐로 생성하는 이유 ? = 각 자릿수에 맞는 큐를 여러개 만들어서 인덱스로 구분할 수 있기 때문에
        ArrayList<Queue<Integer>> list = new ArrayList<>();

        // 1) 0 ~ 9 까지의 인덱스 생성
        for (int i = 0; i < 10; i++) {
            list.add(new LinkedList<>());
        }

        // ② 변수 할당
        int idx = 0;    // arr[]에 재정렬 할 인덱스 값
        int div = 1;    // 각 자릿수 값을 찾기 위해 나눠줄 값
        int maxLen = getMaxLen(arr);    // 최대 자릿수

        // ③ 각 자릿수에 맞는 인덱스번호에 저장
        //  ㄴ 1) 자릿수만큼 반복
        for (int i = 0; i < maxLen; i++) {

            // ㄴ 2) 숫자 값을 하나씩 순회
            for (int j = 0; j < arr.length; j++) {

                // ㄴ 3) (arr[j] / div) % 10 == 제일 끝 자리수
                //      => 제일 끝 자리수 값을 가진 인덱스에 해당 값 저장 (큐이기 때문에 같은 인덱스에 중복가능!)
                list.get( (arr[j] / div) % 10 ).offer(arr[j]);
            }

            // ④ 큐에 넣고 -> 빼면서 arr[]에 재배치
            for (int j = 0; j < 10; j++) {
                Queue<Integer> queue = list.get(j);

                while (!queue.isEmpty()){
                    arr[idx++] = queue.poll();
                }
            }   // ④ 종료

            // ⑤ 다음 자릿수 정렬을 위해 변수값 재할당
            idx = 0;
            div *= 10;

        }   // ③ 종료

    }

    // # 8-1. 최대 자릿수 구하는 메소드
    public static int getMaxLen(int[] arr) {
        int maxLen = 0;

        for (int i = 0; i < arr.length; i++) {
            // ① 자리수 (=len) = 밑이 10인 log 계산값에 + 1
            // ex ) arr[i] = 10 , log10 = 1 , 1 + 1 = 2 => 자릿수 = 2
            int len = (int) Math.log10(arr[i]) + 1;

            // ② 최대 자릿수 저장
            if (maxLen < len){
                maxLen = len;
            }
        }

        return maxLen;
    }


    public static void main(String[] args) {
        // Test code
        int[] arr = {10, 32, 52, 27, 48, 17, 99, 56};
        radixSort(arr);
        System.out.println("기수 정렬: " + Arrays.toString(arr));

    }
}
