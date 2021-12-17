//date: 2021-12-17T16:54:10Z
//url: https://api.github.com/gists/20658d8ca2b1b35b328598092f28b4a3
//owner: https://api.github.com/users/ivtkac

class Main {
    public static void main(String[] args) {
        int[] arr = new int[]{-3, 5, -7, 2, 1, 0};
        sort(arr, 0, arr.length - 1);
        for (int index = 0; index < arr.length; index++) {
            System.out.println(arr[index]);
        }
    }

    public static void swap(int[] numbers, int firstIndex, int secondIndex) {
        int temp = numbers[firstIndex];
        numbers[firstIndex] = numbers[secondIndex];
        numbers[secondIndex] = temp;
    }

    public static int partition(int[] numbers, int begin, int end) {
        int pivot = numbers[end];
        int index = begin - 1;
        for (int i = begin; i <= end - 1; i++) {
            if (numbers[i] < pivot) {
                index++;
                swap(numbers, index, i);
            }
        }
        swap(numbers, index + 1, end);
        return index + 1;
    }

    public static void sort(int[] numbers, int begin, int end) {
        if (begin < end) {
            int partitionIndex = partition(numbers, begin, end);
            sort(numbers, begin, partitionIndex - 1);
            sort(numbers, partitionIndex + 1, end);
        }
    }
}