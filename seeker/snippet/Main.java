//date: 2022-10-18T17:13:56Z
//url: https://api.github.com/gists/48a405f9214236af90b3857ebd663ade
//owner: https://api.github.com/users/lubaochuan

/* lessons learned:
- Test the base cases
- Use multiple debugging code and sources
- Make sure to think of other base cases or problems beforehand on the input side outside of the test cases given
*/

import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

class Main {
  public static void main(String[] args) throws FileNotFoundException{
    //Scanner scan = new Scanner(new File("input6.txt"));
    Scanner scan = new Scanner(System.in);
    while(scan.hasNext()) {
      int start = scan.nextInt();
      int stop = scan.nextInt();
      Boolean switched = false;
      if(start > stop){
        int junk = start;
        start = stop;
        stop = junk;
        switched = true;
      }
      int s = start;
     
      int n = start;
      int max = 1;

      while(start<=stop)
      {
        n = start;
        int cycle = 1;
        while(n>1){
          if (n%2==0)
            n/=2;
          else{
            n=n*3+1;

          }
          cycle++;
 
          if (max<cycle)
            max = cycle;
        }
        start++;
      }
      //System.out.println("max_n:"+max_n);
     
      if(switched){
        System.out.println(stop + " " + s + " " + max);
      } else {
        System.out.println(s + " " + stop + " " + max);
      }
    }
  }
}