//date: 2022-09-01T16:56:59Z
//url: https://api.github.com/gists/ea299793c259e545246b64ce2777ffe7
//owner: https://api.github.com/users/delta-dev-software

interface Hello{
    void sayHello();
}
public class HelloWorld{

     public static void main(String []args){
       Hello hello=()->{
           System.out.println("hello world !");
       };
       hello.sayHello();
     }
}