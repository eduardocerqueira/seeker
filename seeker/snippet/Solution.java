//date: 2023-04-04T17:05:13Z
//url: https://api.github.com/gists/94acce2c9ab67c18dcd7f09348324a97
//owner: https://api.github.com/users/jpalvesloiola

//Complete this code or write your own from scratch
import java.util.*;
import java.io.*;

class Solution{
	public static void main(String []argh){
        Map<String, Integer> phoneBook = new HashMap<>();
		Scanner in = new Scanner(System.in);
		
        int n = in.nextInt(); 
		in.nextLine();
        //save the phone book
		for(int i = 0; i < n; i++){
			phoneBook.put(in.nextLine(), in.nextInt());
			in.nextLine();
		}   
        
        //do the query
		while(in.hasNext()){
            String name;
            name = in.nextLine();
            if (phoneBook.containsKey(name)) {
                System.out.println(name + "=" + phoneBook.get(name));
            } else {
                System.out.println("Not found");
            }	
		}
        
        in.close();
	}
}
