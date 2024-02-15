//date: 2024-02-15T17:02:51Z
//url: https://api.github.com/gists/250203f7454558821d1d05d1551cf3c9
//owner: https://api.github.com/users/ayush-crio

import java.util.*;

// structure of every node
class Trie {
    Trie[] children = new Trie[2];

    Trie() {
        for(int i = 0; i < 2; i++)
            children[i] = null;
    }

}

public class MaximumXor {
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in); 
        int n;
        n = sc.nextInt();
        ArrayList<Integer> a = new ArrayList<>();
        for(int i=0;i<n;i++){
            a.add(sc.nextInt());
        }
        int ans = maximumXor(n,a);
        System.out.println(ans);
        sc.close();
    }
    // Implement Your Solution Here
    public static int maximumXor(int n, ArrayList<Integer> nums) {
        if(nums == null || nums.size() == 0) {
            return 0;
        }
        
        
        
        
        // Initialise my  Trie.
        Trie root = new Trie();
        // iterate over the numbers one by one
        for(int num: nums) {
            // insert method
            Trie curNode = root;
            for(int i = 31; i >= 0; i --) {
                // this checks if the ith bit is set or unset
                int curBit = (num >>> i) & 1; 
                if(curNode.children[curBit] == null) {
                    curNode.children[curBit] = new Trie();
                }
                curNode = curNode.children[curBit];
            }
        }



        int max = Integer.MIN_VALUE;
        for(int num: nums) {

            Trie curNode = root;
            
            int curSum = 0;
            for(int i = 31; i >= 0; i --) {
                int curBit = (num >>> i) & 1;
                // currBit is set, (curBit ^ 1) -> unset value
                // currBit is unset, (curBit ^ 1) -> set value
                if(curNode.children[curBit ^ 1] != null) {
                    curSum += (1 << i);
                    curNode = curNode.children[curBit ^ 1];
                }else {
                    curNode = curNode.children[curBit];
                }
            }
            max = Math.max(curSum, max);
        }
        return max;
    }    
}
