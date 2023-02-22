//date: 2023-02-22T16:52:30Z
//url: https://api.github.com/gists/89062d34bdd72702f790a62109d6dffd
//owner: https://api.github.com/users/kumarsumiit

public class reverse {
    public static void main(String[] args) {
        int[]arr = {9,7,3,4,5};
        int l = arr.length;
        int n =Math.floorDiv(l,2);
        int temp;

        for(int i=0; i<n; i++){
            //swap a[i] and a[l-i-1] (where l = length)
            //  arr i   b   temp
            // |4|     |3|   ||  ===> |3| || |4| ===> |3| |4| ||
            temp = arr[i];
            arr[i] = arr[l-i-1];
            arr[l-i-1] = temp;

        }
        for(int element:arr){
            System.out.print(element + " ");
        }
    }
}
