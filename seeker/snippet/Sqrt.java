//date: 2025-12-25T17:06:03Z
//url: https://api.github.com/gists/a17aa58a85177a1aed62282fda2da4bb
//owner: https://api.github.com/users/samarthsewlani

class Solution {
    public int mySqrt(int x) {
        long low=0,high=x;
        if(high*high==x)    return (int)high;
        while(low<high){
            long middle=(long)((high+low)/2);
            if(low==middle) return (int)middle;
            if(middle*middle<x){
                low=middle;
            }
            else if(middle*middle>x){
                high=middle;
            }
            else if(middle*middle==x)   return (int)middle;
        }
        return (int)low;
    }
}