//date: 2024-08-26T16:52:44Z
//url: https://api.github.com/gists/334da33b328a058c4a5bacd60a2340cf
//owner: https://api.github.com/users/RamshaMohammed

class thread extends Thread
{
    public void run()
    {
        try {
            for(int i=1;i<=7;i++)
            {
                System.out.println("Ramsha");
                Thread.sleep(1000);
            }
        }
        catch(InterruptedException e)
        {
            System.out.println("I like to explore new places");
        }
    }
}
public class Interrupt {
    public static void main(String[] args)
    {
        thread t1 = new thread();
        t1.start();
        t1.interrupt();
        System.out.println(t1.isInterrupted());
        System.out.println(t1.interrupted());
        for(int i = 0;i<=6;i++)
        {
            System.out.println("I am from vijayawada"+i);
            
       }
   }
}