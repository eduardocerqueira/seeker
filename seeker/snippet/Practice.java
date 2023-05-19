//date: 2023-05-19T16:41:15Z
//url: https://api.github.com/gists/d47a195caf734754a637670e84584b7f
//owner: https://api.github.com/users/yeon-ju-k

// Practice 1
// 종이 접기
// 종이를 반으로 접었을 때, 안으로 파인 부분은 0, 볼록 튀어나온 부분은 1이라고 하자.
// 종이를 접을 때는 오른쪽에서 왼쪽으로 접는다.
// 종이를 N번 접었을 때의 접힌 상태를 출력하는 문제를 작성하세요.

// 입출력 예시
// 입력 : 1
// 출력 : 0

// 입력 : 2
// 출력 : 0, 0, 1

// 입력 : 3
// 출력 : 0, 0, 1, 0, 0, 1, 1

public class Practice {

    public static void solution (int n){

        // 트리 생성
        // ㄴ 생성되는 트리가 포화이진트리 이므로
        //    -> 해당 트리의 노드의 개수(사이즈)는 : 2의 (높이+1)승 - 1개
        //    -> 높이는 : n - 1
        int[] binaryTree = new int[(int) Math.pow(2,n) -1];

        // 처음 접었을 때
        binaryTree[0] = 0;

        // 나머지 횟수만큼 접을때
        // ㄴ 마지막 부모의 인덱스까지 for 문 돌리기
        // ㄴ> (인덱스가 1부터일때) 자식노드의 부모인덱스 번호는 = 자식인덱스번호 / 2 이다.
        // ㄴ> (응용해서 인덱스가 0일때) 마지막 부모의 인덱스 번호 = ((마지막 인덱스번호+1) / 2) - 1)
        int lastPIdx = binaryTree.length / 2 - 1;
        for (int i = 0; i <= lastPIdx; i++) {
            // 왼쪽 자식노드에 0값을, 오른쪽 자식노드에 1값 넣기
            int left = (2 * i) + 1;
            int right = (2 * i) + 2;
            binaryTree[left] = 0;
            binaryTree[right] = 1;
        }   // for 문 종료

        // 중위 순회 메소드 출력
        inOrder(binaryTree, 0);
        System.out.println();

    }

    // 중위 순회로 출력 ( 왼 -> 현재노드 -> 오 )
    public static void inOrder(int[] arr, int idx){

        // 왼쪽 자식 노드 출력
        int left = 2 * idx + 1;
        if (left < arr.length){
            // 왼쪽 자식노드를 현재노드로 하는 inOrder 재귀
            inOrder(arr, left);
        }

        // 현재 노드 출력
        System.out.print(arr[idx] + " ");

        // 오른쪽 자식 노드 출력
        int right = 2 * idx + 2;
        if (right < arr.length){
            // 오른쪽 자식노드를 현재노드로 하는 inOrder 재귀
            inOrder(arr, right);
        }

    }


    public static void main(String[] args) {

        solution(1);
        solution(2);
        solution(3);

    }
}
