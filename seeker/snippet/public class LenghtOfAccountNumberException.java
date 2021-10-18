//date: 2021-10-18T17:06:25Z
//url: https://api.github.com/gists/ad0ad6e6e5543415debb135712346621
//owner: https://api.github.com/users/anelaco

public class LenghtOfAccountNumberException extends Exception {

    public LenghtOfAccountNumberException(){
        System.out.println("This is not okay");
    }

    public LenghtOfAccountNumberException(String message) {
        super(message);
    }
}
