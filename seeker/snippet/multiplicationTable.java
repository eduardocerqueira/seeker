//date: 2023-12-01T16:42:19Z
//url: https://api.github.com/gists/b168fb582bbe8f12eed30ebade2ccf17
//owner: https://api.github.com/users/youngsik823

package project;

public class multiplicationTable {
  public static void main(String[] args) {
    System.out.println("[구구단 출력]");
    for (int i = 1; i <= 9; i++) {
      for (int j = 1; j <= 9; j++) {
        System.out.printf(String.format("%02d x %02d = %02d     ", j, i, i*j));
      }
      System.out.println();
    }
  }
}

// 윤영식
