//date: 2024-08-26T16:50:46Z
//url: https://api.github.com/gists/70d4b0678c1093f6ac2ffe5f2bd24005
//owner: https://api.github.com/users/RamshaMohammed

class Metals1 extends Thread
{
    public void run()
    {
        for(int i =0;i<=7;i++)
        {
            System.out.println("Shine bright like a Diamond"+(i+1));
        }
    }
}
public class MultiThreading {
    public static void main(String[] args)
    {
        Metals1 t1 = new Metals1();
        System.out.println(t1.getName());
        t1.setName("Ramsha knows different metals");
        System.out.println(t1.getName());
        System.out.println(t1.getPriority());
        t1.setPriority(7);
        System.out.println(t1.getPriority());
        t1.start();
        
        Metals2 t2 = new Metals2();
        Thread t = new Thread(t2);
        System.out.println(t.getName());
        t.setName("Ramsha has silver ring");
        System.out.println(t.getName());
        System.out.println(t.getPriority());
        t.setPriority(7);
        System.out.println(t.getPriority());
        t.start();
        for(int i=0;i<=7;i++)
        {
            System.out.println("i like Diamond ring"+(i+1));
       }
   }
}
class Metals2 implements Runnable
{
    public void run()
    {
        for(int i =0;i<=7;i++)
        {
            System.out.println("Old is Gold "+(i));
        }
    }
}