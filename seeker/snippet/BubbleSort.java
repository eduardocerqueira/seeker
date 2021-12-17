//date: 2021-12-17T17:00:52Z
//url: https://api.github.com/gists/22f74210d2bd57684ba78b52b0e4bffd
//owner: https://api.github.com/users/ivtkac

class Main {
    public static void main(String[] args) {
        int[] arr = new int[]{-3, 5, -7, 2, 1, 0};
        sort(arr);
        for (int index = 0; index < arr.length; index++) {
            System.out.println(arr[index]);
        }
    }

    public static void sort(int[] numbers) {
        int length = numbers.length;
        for (int i  = 0; i < length - 1; i++) {
            for (int j = length - 1; j > i; j--) {
                if (numbers[j] < numbers[j - 1]) {
                    int temp = numbers[j - 1];
                    numbers[j - 1] = numbers[j];
                    numbers[j] = temp;
                }
            }
        }
    }
}