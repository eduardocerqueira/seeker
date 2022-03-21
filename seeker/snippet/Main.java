//date: 2022-03-21T17:06:00Z
//url: https://api.github.com/gists/641e9e375721ba678c85eccf768b9027
//owner: https://api.github.com/users/ramya961

class Main implements Runnable{
    public void run(){
        System.out.println("Current Thread Name is " + Thread.currentThread().getName());
    }
public static void main(String args[]){
    Main n1=new Main();
    Thread t1=new Thread(n1);
    t1.setName("NewThread");
    System.out.println(Thread.currentThread().getName());
    System.out.println("Threading is about to start");
    t1.start();
}}