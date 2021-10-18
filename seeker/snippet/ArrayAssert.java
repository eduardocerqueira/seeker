//date: 2021-10-18T17:06:25Z
//url: https://api.github.com/gists/ad0ad6e6e5543415debb135712346621
//owner: https://api.github.com/users/anelaco

public class ArrayAssert {

    /**
     * Implementation of bubble sort
     * @param ar unsorted array
     * @return sorted array
     */
    public static int[] sort(int[] ar){
        assert (!isSorted(ar));
        for(int i = 0; i < ar.length-1; i++){
            for(int j = 0; j < ar.length -1 -i; j++ ){
                if(ar[j]> ar[j+1]){
                    int temp = ar[j];
                    ar[j] = ar[j+1];
                    ar[j+1] = temp;
                }
            }
        }
        return ar;
    }

    /**
     * @param ar unsorted array in a increasing manner
     * @return true if array is sorted
     */
    public static boolean isSorted(int[] ar){
        if(ar.length == 0 || ar.length == 1)
            return true;
        for(int i = 1; i < ar.length;i++){
            if(ar[i -1] > ar[i])
                return false;
        }
        return true;
    }

    public static void main(String [] args) {
        int[] a = {1,3,5,7, 428, 24,5,4,68,843,1};
        assert (!isSorted(a)); 
        a = sort(a);
        for(int i = 0; i < a.length; i++){
            System.out.print(a[i] + ", ");
        }

    }
}
