//date: 2022-04-21T17:05:47Z
//url: https://api.github.com/gists/977445555da45a8f4f70e7e6752ebe42
//owner: https://api.github.com/users/gen-eslam

//https://codeforces.com/contest/746/problem/B
//B. Decoding
import java.util.*;

public class codeforces {
    public static void main(String[] arg) {
        Scanner scanner = new Scanner(System.in);
        LinkedList<String> linkedList =new LinkedList<>();

        int x =scanner.nextInt();
        char[] arr = scanner.next().toCharArray();
       if(x<=2)
       {
           for(int index =0;index<x;index++)
           {
               System.out.print(arr[index]);


           }


       }else
       {
           linkedList.add( String.valueOf(arr[0]));
           for(int index =1;index<arr.length;index++)
           {
               if(index%2 !=0)
                   linkedList.addFirst( String.valueOf(arr[index]));
               else
                   linkedList.addLast( String.valueOf(arr[index]));

           }
          if(x%2 ==0){
              for(int index =linkedList.size()-1;index>=0;index--)
              {
                  System.out.print(linkedList.get(index));


              }

          }else { for(int index =0;index<x;index++)
          {
              System.out.print(linkedList.get(index));


          }}
       }






    }
}
