//date: 2024-09-04T17:09:01Z
//url: https://api.github.com/gists/aff7df898f954f2e5bfb59f1da69f51f
//owner: https://api.github.com/users/sasub-mlp

public class forty_five {
    static int remove_duplicate(int[] arr,int n){
        if (n==0 || n==1){
            return n;
        }
        int j=0;
        for (int i=0;i<n-1;i++){
            if (arr[i]!=arr[i+1]){
                arr[j++]=arr[i];
            }
        }
        arr[j++]=arr[n-1];
        return j;
    }
    public static void main(String[] args){
        int[] arr = {1, 2, 2, 3, 4, 4, 4, 5, 5};
        int n = arr.length;
        System.out.println("Before removal: ");
        for (int i=0;i<n;i++){
            System.out.print(arr[i]+" ");
        }
        n=remove_duplicate(arr,n);
        System.out.println("\nAfter removal: ");
        for (int i=0;i<n;i++){
            System.out.print(arr[i]+" ");
        }
    }
}
