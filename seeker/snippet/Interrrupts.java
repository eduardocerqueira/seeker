//date: 2024-08-15T16:45:57Z
//url: https://api.github.com/gists/618d6af1eb71663d59b2c2d6e35cbc5f
//owner: https://api.github.com/users/ShobharaniPotru

 class Interrrupts implements Runnable
{
    Thread t;
    public Interrrupts()
    {
        t = new Thread(this);
        System.out.println("Executing" + t.getName());
        t.start();
        if (!t.interrupted())
        {
            // Interrupts this thread.
            t.interrupt();
        }
        // block until other threads finish
        try
        {
            t.join();
        }
        catch (InterruptedException e)
        {

        }
    }
    public void run()
    {
        try
        {
            while (true)
            {
                Thread.sleep(1000);
            }
        }
        catch (InterruptedException e)
        {
            
            System.out.println(t.getName() + "interrupted:");
            System.out.println(e.toString() + "\n");
        }
    }
}
public class ThreadDE
{
    public static void main(String[] args) {
        new Interrrupts();
        new Interrrupts();
    }
}