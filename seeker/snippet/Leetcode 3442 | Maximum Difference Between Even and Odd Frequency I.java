//date: 2025-06-10T17:03:19Z
//url: https://api.github.com/gists/583d253d461c640bc7d8c7b5eeb5a49b
//owner: https://api.github.com/users/Tan12d

class Solution {
    public int maxDifference(String s) 
    {
        int freq[] = new int[26];
        
        for(char c: s.toCharArray())
        {
        	freq[c-'a']++;
        }
        
        int minEvenFreq = Integer.MAX_VALUE;
        int maxOddFreq = Integer.MIN_VALUE;
        
        for(int i: freq)
        {
        	if((i&1)==0 && i!=0 && i<minEvenFreq)
        	{
        		minEvenFreq=i;
        	}
        	
        	if((i&1)==1 && i>maxOddFreq)
        	{
        		maxOddFreq=i;
        	}
        }
                
        return maxOddFreq-minEvenFreq;        
    }
}