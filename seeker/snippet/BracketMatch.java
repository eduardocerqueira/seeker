//date: 2023-01-02T17:10:13Z
//url: https://api.github.com/gists/29e327d8301a4341d239eef6a819b294
//owner: https://api.github.com/users/ridhofauza

import java.util.*;

public class BracketMatch {
   public static int bracket_match(String bracket_string) {
      Stack<Character> stack1 = new Stack<>();
      Stack<Character> stack2 = new Stack<>();
      for(int i = 0; i < bracket_string.length(); i++) {
         char charBracket = bracket_string.charAt(i);
         if (charBracket == '(') {
            stack1.push(charBracket);
         } else {
            if (stack1.size() > 0) {
               stack1.pop();
            } else {
               stack2.push(charBracket);
            }
         }
      }
      return (stack1.size()+stack2.size());
   }

   public static void main(String[] args) {
      String str = "(()())";
      // String str = "((())";
      // String str = "())))";
      // String str = ")()(";
      // String str = "))(((";
      // String str = "((())";
      // String str = "())";
      // String str = ")(";

      System.out.println(bracket_match(str));
   }
}