//date: 2025-07-23T16:58:59Z
//url: https://api.github.com/gists/feedc1d37836ba8e760817d4e9c083c7
//owner: https://api.github.com/users/IshuAndani

class Solution {
    public int subarraysWithKDistinct(int[] nums, int k) {
        ExecutorService executor = Executors.newFixedThreadPool(2);

        Callable<Integer> task1 = () -> countAtMost(nums, k);
        Callable<Integer> task2 = () -> countAtMost(nums, k - 1);
        
        Future<Integer> future1 = executor.submit(task1);
        Future<Integer> future2 = executor.submit(task2);

        int result = -1;

        try {
            result = future1.get() - future2.get();
        } 
        catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        } 
        finally{
            executor.shutdown();
        }

        return result;
        //return countAtMost(nums,k) - countAtMost(nums,k-1);
    }

    //private int countAtMost(int nums[], int k){
    //    if(k == 0) return 0;
    //    int l = 0;
    //    int r = 0;
    //    int res = 0;
    //    int map[] = new int[nums.length+1];
    //    int size = 0;
    //    while(r < nums.length){
    //        int f = map[nums[r]];
    //        if(f == 0){
    //            while(size == k && f == 0){
    //                int fl = map[nums[l++]]--;
    //                if(fl == 1){
    //                    size--;
    //                }
    //            }
    //            size++;
    //        }
    //        map[nums[r++]]++;
    //        res += r-l;
    //    }
    //    return res;
    //}
    
    private int countAtMost(int nums[], int k){
        if(k == 0) return 0;
        int l = 0;
        int r = 0;
        int res = 0;
        HashMap<Integer,Integer> map = new HashMap<>();
        while(r < nums.length){
            int f = map.getOrDefault(nums[r],0);
            if(f == 0 && map.size() == k){
                while(true){
                    int fl = map.get(nums[l]);
                    if(fl == 1){
                        map.remove(nums[l++]);
                        break;
                    }
                    map.put(nums[l++],fl-1);
                }
            }
            map.put(nums[r++],f+1);
            res += r-l;
        }
        return res;
    }
}