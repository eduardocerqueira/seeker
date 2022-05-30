//date: 2022-05-30T17:10:21Z
//url: https://api.github.com/gists/83d905028e040dd5ca43bf39aab7ac0f
//owner: https://api.github.com/users/ktam512

public class Main {
    public static void main(String[] args){
        int a = 6;
        int b = 3;
        System.out.println("a+b= " +(a+b));
        System.out.println("a-b= " +(a-b));
        System.out.println("b-a= " +(b-a));
        System.out.println("a*b= " +(a*b));
        System.out.println("a/b= " +(a/b));
        System.out.println("b/a= " +(b/a));

        int i =1;
        i++;
        System.out.println("Với i++ thì i là " + i);
        int c =1;
        ++c;
        System.out.println("Với ++c thì c là " + c);

        System.out.println("Vậy ta có i++ và ++i là " + ((i==c)? "tương tự":"khác nhau"));
    }
}
