//date: 2024-08-26T16:46:54Z
//url: https://api.github.com/gists/415476a667aad137ad41f7de8877aae7
//owner: https://api.github.com/users/RamshaMohammed

public class Left_Right_String_Rotations
{
    public static String Left(String str, int n)
    {
        int len = str.length();
        n = n % len;
        return str.substring(n) + str.substring(0,n);
    }
    public static String Right(String str, int n)
    {
        int len = str.length();
        n = n % len;
        return str.substring(len - n) + str.substring(0,len-n);
   }
    public static void main(String[] args)
    {
        String str = "Ramsha";
        String Left = Left(str,2);
        System.out.println("Left String is: "+Left);
        String Right = Right(str,2);
        System.out.println("Right String is: "+Right);
    }
}