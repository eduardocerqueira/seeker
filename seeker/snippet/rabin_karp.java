//date: 2022-02-18T16:51:42Z
//url: https://api.github.com/gists/2fefc83a03d67bab780aa06f48771557
//owner: https://api.github.com/users/J-Ankit2020

package strings;

public class rabin_karp {
    public static final int d = 256;
    //    it is the number of characters in the string generally a prime number
    static void search(String str,String pattern,int q){
        int t = 0; // it will store hash value of str window
        int p = 0; // will store hash value of pattern
        int h = 1;
        int m = pattern.length();
        int n = str.length();
//        h will be the pow(d,m-1)
        for(int i=0;i<m-1;i++){
            h = (h*d)%q;
        }
//        calculating the num value of first window of str and pattern
        for(int i=0;i<m;i++){
            p = (p*d+pattern.charAt(i))%q;
            t = (t*d+str.charAt(i))%q;
        }
        for (int i = 0; i <= n-m; i++) {
            if (t==p){
                int j = 0;
                for (; j < m; j++) {
                    if (str.charAt(i+j)!=pattern.charAt(j)){
//                        this means that string are not identical
                        break;
                    }
                }
                if (j==m) {
                    // this means we found one valid pattern
                    System.out.println("Pattern found at index " + i);
                }
            }
            if (i<n-m){
//                if i is less than n-m than only we can go ahead
                t = (d * (t - str.charAt(i)*h) + str.charAt(i+m)) % q;
                if (t<0){
                    t = t + q;
                }
            }
        }
    }
    public static void main(String[] args) {
        String txt = "GEEKS FOR GEEKS";
        String pat = "GEEK";
        int q = 101;

        // Function Call
        search(txt,pat, q);
    }
}
