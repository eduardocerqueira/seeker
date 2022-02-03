//date: 2022-02-03T17:01:33Z
//url: https://api.github.com/gists/920afac80030073dcf484e2daf01fe10
//owner: https://api.github.com/users/neer2808


 class ArrayAssign10 {
  public static int[] check(int[] nums) {
    // if there are only two elements in the array then return the array
    if (nums.length < 2) return nums;
    // result array with same length
    int[] result = new int[nums.length];
    // here use two variables using which i am arranging by5 values at right side and
    // notby5 values at left side
    // notby5 values will arrange from index o and odd values will arrange from
    int by5 = 0;
    int notby5 = nums.length - 1;

    for (int i = 0; i < nums.length; i++) {
      if (nums[i] % 5 == 0)
        result[notby5--] = nums[i];
      else

      result[by5++] = nums[i];
    }
    return result;
  }
  public static void main(String[] args) {
    int arr[] = {10,30,23,5,6};
    int rarr[]=  check(arr);
    for (int i = 0; i <rarr.length ; i++) {
      System.out.println(rarr[i]);
    }
  }
}