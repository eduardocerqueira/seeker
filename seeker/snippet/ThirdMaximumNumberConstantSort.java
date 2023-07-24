//date: 2023-07-24T16:47:31Z
//url: https://api.github.com/gists/265e5b8f26b0c4cbe7f698cba146b5ec
//owner: https://api.github.com/users/Andres-Fuentes-Encora

class ThirdMaximumNumberConstantSort {
    public int thirdMax(int[] nums) {
        Integer[] arr = new Integer[4];
        Set<Integer> set = new HashSet<>();
        int index = 0;
        for(int num : nums){
            if(set.contains(num)){
                continue;
            }
            set.add(num);
            arr[3] = num;
            Arrays.sort(arr, (a,b) -> {
                if(a == null && b == null){
                    return 0;
                } else if(a == null) {
                    return 1;
                } else if(b == null){
                    return -1;
                } else {
                    return a < b ? 1 : -1;
                }
            }); 
        }

        if(arr[2] == null){
            return arr[0];
        } else {
            return arr[2];
        }
    }
}