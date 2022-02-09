//date: 2022-02-09T17:14:51Z
//url: https://api.github.com/gists/1f26544c2df1c9b7e1094f4d154fc5b6
//owner: https://api.github.com/users/NamanSaini18

// permutation example
// Example: the input string is "abc"
// print all the permutations of the given string
//
// abc
// acb
// bac
// bca
// cab
// cba
public class Permutation {
    public static void answer(String str, String ans){
        // Base condition
        if(str.length()==0){
            System.out.println(ans);
            return;
        }
        for (int i = 0; i < str.length(); i++) {
            String Remaining_String = str.substring(0,i)+ str.substring(i+1);
            char extract = str.charAt(i);
            answer(Remaining_String,ans+extract);    //Recursive call
            
        }
    }

    public static void main(String[] args) {
        answer("8868"," ");
    }
}
