//date: 2025-12-25T17:07:29Z
//url: https://api.github.com/gists/8acaf16871305ff20b294dca0a7fdaa0
//owner: https://api.github.com/users/samarthsewlani

class Solution {
    public int[] sortByBits(int[] arr) {
        List<Integer> list = new ArrayList<>();
        for(int x : arr)        list.add(x);
        Collections.sort(list, (a, b) -> countBitOne(a) == countBitOne(b) ? a - b : countBitOne(a) - countBitOne(b));
        for(int i=0;i<arr.length;i++)       arr[i] = list.get(i);
        return arr;
    }
    private int countBitOne(int n) {
        int res = 0;
        while (n != 0) {
            res += (n & 1);
            n >>= 1;
        }
        return res;
    }
}