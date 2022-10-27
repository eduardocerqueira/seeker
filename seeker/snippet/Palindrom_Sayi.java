//date: 2022-10-27T17:21:05Z
//url: https://api.github.com/gists/f2dc33253e4038a0af32e4b1fb3234ee
//owner: https://api.github.com/users/toprakcobanoglu

package java101.metot;

public class Palindrom_Sayi {
    static boolean isPalindrom(int number)  {
        int temp = number, reverseNumber = 0, lastNumber;
        while (temp != 0)   {
            lastNumber = temp % 10;
            reverseNumber = (reverseNumber * 10) + lastNumber;
            temp /= 10;
        }
        if (number == reverseNumber)
            return true;
        else
            return false;
    }
    public static void main(String[] args) {
        System.out.println(isPalindrom(890));
    }
}
