//date: 2025-02-21T17:01:11Z
//url: https://api.github.com/gists/6e0380d449c0de47a9e119dc24cc8a56
//owner: https://api.github.com/users/icarito

import java.util.ArrayList;

class Solution{
    private static boolean isPalindrome(int num) {
        String s = Integer.toString(num);
        return s.equals(new StringBuilder(s).reverse().toString());
    }
    public static int values (int n){
      ArrayList<Integer> validPalindromes = new ArrayList<>();
      // recorrer todos los números desde 1 hasta n
      for (int a = 1; a * a < n; a++) {
        
        // calcular desde el numero a hasta n
        // las posibles sumas de cuadrados
        int sum = a * a;
        int next = a + 1;
        while (true) {
          sum += next * next;
          if (sum >= n) {
            break;
          }
          // chequeamos si son palíndromos
          if (isPalindrome(sum)) {
            // si son palíndromos los añadimos a un ArrayList (para evitar duplicados)
            if (!validPalindromes.contains(sum)) {
              validPalindromes.add(sum);
            }
          }
          next++;
        }
      }
      return validPalindromes.size();
    }
}