//date: 2022-01-27T16:59:49Z
//url: https://api.github.com/gists/54abe4c4b47600ee07572670d80f52e3
//owner: https://api.github.com/users/akshay-ravindran-96

class Solution {
    public boolean containsDuplicate(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        
        for(int i: nums)
        {
            if(map.containsKey(i))
                return true;
            else
                map.put(i, 1);
        }
        
        return false;
        
    }
}