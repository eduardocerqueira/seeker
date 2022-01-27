//date: 2022-01-27T17:08:55Z
//url: https://api.github.com/gists/335c9d542a3c0d83a3381f642e5e33b8
//owner: https://api.github.com/users/akshay-ravindran-96

class Solution {
    public List<List<Integer>> threeSum(int[] arr) {
        
        List<List<Integer>> ans = new ArrayList <List<Integer>>();
        List<Integer> row = new ArrayList<Integer>();
            TreeSet<String> t = new TreeSet<String>();
    
        Arrays.sort(arr);
        int l;
        int r;
        for(int i=0; i<arr.length-2; i++)
        {
            l=i+1;
            r=arr.length-1;
        while(l<r)
        {
            if(arr[i] +arr[l] +arr[r] == 0)
            {    
                 String str = arr[i] +":"+ arr[l] + ":"+ arr[r] ;
                    if(!t.contains(str))
                    {
                        row.add(arr[i]);
                         row.add(arr[l]);
                        row.add(arr[r]);
                        ans.add(row);
                          row= new ArrayList<>();
                       
                        t.add(str);
                    }
           
             l++;
             r--;
            }
           
           else if(arr[i] +arr[l] +arr[r] > 0)
            {
                r--;
            }
            else
            {
                l++;
            }   
        }   
        }
        return ans;
        
    }
}