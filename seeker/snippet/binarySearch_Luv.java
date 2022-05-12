//date: 2022-05-12T17:19:37Z
//url: https://api.github.com/gists/d73f765c18b9ee379eaaabb024a41d13
//owner: https://api.github.com/users/RavikantSa

package com.pepcoding;
class fisOcc{
    int[] arr;
    int n;
    int key;
   void  firstOcc( int [] arr , int n, int key) {

        int s = 0, e = n-1;
        int mid = s + (e-s)/2;
        int ans = -1;
        while(s<=e) {

            if(arr[mid] == key){
                ans = mid;
                e = mid - 1;
            }
            else if(key > arr[mid]) {//Right me jao
                s = mid + 1;
            }
            else if(key < arr[mid]) {//left me jao
                e = mid - 1;
            }

            mid = s + (e-s)/2;
        }
        System.out.println(ans);
    }
}
class lastOcc {
    int[] arr;
    int n;
    int key;
   void lastOcc(int [] arr, int n, int key) {

        int s = 0, e = n - 1;
        int mid = s + (e - s) / 2;
        int ans = -1;
        while (s <= e) {

            if (arr[mid] == key) {
                ans = mid;
                s = mid + 1;
            } else if (key > arr[mid]) {//Right me jao
                s = mid + 1;
            } else if (key < arr[mid]) {//left me jao
                e = mid - 1;
            }

            mid = s + (e - s) / 2;
        }
        System.out.println(ans);
    }
}
public class binarySearch_Luv {
    public static void main(String[] args) {
        int [] a = new int []{1,1,1,1,1,2,3,3,3,3,3,3,3,3,3,4};
        fisOcc f = new fisOcc();
        f.firstOcc(a,a.length,3);
        lastOcc l= new lastOcc();
        l.lastOcc(a,a.length,3);


    }
}
