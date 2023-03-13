//date: 2023-03-13T16:49:12Z
//url: https://api.github.com/gists/ee8346a5412a9672e313e025ea9ca41b
//owner: https://api.github.com/users/shourav9884

// Problem 1
private Integer[] problem1(Integer[] arr) {
    Integer[] result = new Integer[arr.length];
    int j=0;
    for (int i=0;i<arr.length;i++) {
        if (arr[i] != 0) {
            result[j] = arr[i];
            j++;
        }
    }
    for (;j<arr.length;j++) {
        result[j] = 0;
    }
    return result;
}
// Problem 1 (Optimized)
private Integer[] optimalProblem1(Integer[] arr) {
    // 0,1,0,3,12
    int j=0;
    for (int i=0;i<arr.length;i++) {
        if (arr[i] != 0) {
            arr[j] = arr[i];
            j++;
        }
    }
    // 1, 3, 12, 3, 12
    for (;j<arr.length;j++) {
        arr[j] = 0;
    }
    // 1, 3, 12, 0, 0
    return arr;
}
// Problem 2
private String[] problem2(int num) {
    String[] result = new String[num];
    for (int i=0;i<num;i++) {
        if ((i+1) % 3 == 0 && (i+1) % 5 == 0) {
            result[i] = "Airwrk";
        } else if ((i+1) % 3 == 0) {
            result[i] = "Air";
        } else if ((i+1) % 5 == 0) {
            result[i] = "wrk";
        } else {
            result[i] = (i+1)+"";
        }
    }
    return result;
}

// Problem 3

private Integer reverseDigit(Integer num) {
    // 123 => 321
    // 123 /10 = 12, 3
    // 12/10 = 1, 2
    // 1/10 = 0, 1
    Integer result = 0;
    while (num != 0) {
        result = result *10 +(num % 10);
        num = num /10;
    }
    return result;
}

private Integer problem3(Integer[] nums) {
    Set<Integer> numSet = new HashSet<Integer>();
    for (int i=0;i<nums.length;i++) {
        numSet.add(nums[i]);
        numSet.add(this.reverseDigit(nums[i]));
    }
    return numSet.size();
}