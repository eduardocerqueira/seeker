//date: 2023-11-17T16:35:57Z
//url: https://api.github.com/gists/a9364e3472edd5daf7239779dacd1f34
//owner: https://api.github.com/users/theaman05

import java.util.*;

class UserDefinedException extends Exception{
    UserDefinedException(String str){
        super(str);
    }
}

public class FactorialWithException{

    static int getFactorial(int num){
        if(num==0){
            return 1;
        }
        return num*getFactorial(num-1);
    }

    public static void main(String[] args){

        Scanner sc = new Scanner(System.in);
        int num, res;


        try{
            System.out.print("Enter a number: ");
            num = sc.nextInt();
            if(num<0){
                throw new UserDefinedException("Negative number is not allowed!");
            }

            res = getFactorial(num);
            System.out.println(num + "! = " + res);
        }
        catch(InputMismatchException e){
            System.out.println("Enter a valid number!");
        }
        catch(UserDefinedException e){
            System.out.println(e.getMessage());;
        }


    }
}