//date: 2023-03-14T17:14:59Z
//url: https://api.github.com/gists/f2ca18cae6d8a4004d050bf7cb0816d8
//owner: https://api.github.com/users/mahesh504

package numbers;

class Fibonacci{
    public static void main(String[] args)  {
        int n1=0,n2=1,n3;
        System.out.print(n1+" "+n2+" ");
        for(int i =2; i < 10; i++){
            n3 = n1+n2;
            System.out.print(n3+" ");
            n1 =n2;
            n2 =n3;
        }
    }
}