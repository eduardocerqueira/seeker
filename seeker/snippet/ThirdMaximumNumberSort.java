//date: 2023-07-24T16:58:00Z
//url: https://api.github.com/gists/2ff155166febbef78020fd0e7d3b3e02
//owner: https://api.github.com/users/Andres-Fuentes-Encora

class ThirdMaximumNumberSort {
    public int thirdMax(int[] nums) {
        Set<Integer> set = new HashSet<>(); 

        for (int i = 0; i < nums.length; i++) {
            set.add(nums[i]);
        }

        List<Integer> cleanedList = new ArrayList<>(set);
        Collections.sort(cleanedList, Collections.reverseOrder());
        if(cleanedList.size() > 2){
            return cleanedList.get(2);
        } else {
            return cleanedList.get(0);
        }
    }
}