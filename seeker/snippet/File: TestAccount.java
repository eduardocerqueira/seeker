//date: 2023-01-31T16:52:24Z
//url: https://api.github.com/gists/f206819d156e95e288f67578e0d056b4
//owner: https://api.github.com/users/Vishuu005

//A Java class to test the encapsulated class Account.  
public class TestEncapsulation {  
public static void main(String[] args) {  
    //creating instance of Account class  
    Account acc=new Account();  
    //setting values through setter methods  
    acc.setAcc_no(7560504000L);  
    acc.setName("Sonoo Jaiswal");  
    acc.setEmail("sonoojaiswal@javatpoint.com");  
    acc.setAmount(500000f);  
    //getting values through getter methods  
    System.out.println(acc.getAcc_no()+" "+acc.getName()+" "+acc.getEmail()+" "+acc.getAmount());  
}  
}  