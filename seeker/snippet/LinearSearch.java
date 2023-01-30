//date: 2023-01-30T16:47:02Z
//url: https://api.github.com/gists/1f88162a74d9bf20add66ee12b847056
//owner: https://api.github.com/users/akbarsiddique

public class LinearSearch {
    public static int LinearSearch(int number[] ,int key){
        for(int  i =0; i<number.length;i++){
            if(number[i] == key){
                return i;
            }
        }
        return -1;
    }
    public static void main(String[]args){
        int number[]  = {111,23,45,67,89,};
        int key =819;
        int index = LinearSearch(number ,key);
        System.out.println("Index is :"+ index);
    }
}
