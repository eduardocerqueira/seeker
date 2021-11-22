//date: 2021-11-22T17:06:15Z
//url: https://api.github.com/gists/1fe28223698ee48cd9454fc00f0847d5
//owner: https://api.github.com/users/BoruE23

import java.util.*;

public class StudentHuffmanRunner {

    static    ArrayList<Node> aNodes = new ArrayList<Node>();
    static    ArrayList<Character> c = new ArrayList<Character>();
    static   ArrayList<Integer> num = new ArrayList<Integer>();

    public static void main(String[] args) {

        String t = "shesellsseashells";
         findf(t);
        //Step 1
        System.out.println(c);
        System.out.println(num);
        //Step 2 B 
        for(int i=0; i<c.size();i++){
            
           
            
            for(int j=0;j<aNodes.size();j++){
                if(aNodes.get(j).data >= num.get(i)){
                    aNodes.add(j,new Node(num.get(i),c.get(i)));
                   j = aNodes.size();
                }
            }
            
             if(aNodes.size() == 0){
              aNodes.add(new Node(num.get(i),c.get(i)));
            }
            
        }
        
        for(int x=0; x<aNodes.size();x++){
             System.out.println("Node: " + aNodes.get(x).data + " " + aNodes.get(x).c );
        }
        
       

        //Step 3  
        
        //Step 4
        /*
        BinaryTree tree = new BinaryTree();
        tree.root = aNodes.get(0);
        tree.showValuesHuff(tree.root, ""); 
        */

    }//end of main method
    

   
    //Frequency Function
    public static void findf(String t){

        for(int i = 0; i<t.length(); i++){
            String ss = t.substring(i,i+1);
            char spot = ss.charAt(0);
            boolean add = true;

            for(int j = 0; j< c.size(); j++){
                if(c.get(j).equals(spot)){
                    add = false;
                    num.set(j,num.get(j) + 1);
                }
            } 

            if(add==true){
                c.add(spot);
                num.add(1);
            }

        }

    }
}