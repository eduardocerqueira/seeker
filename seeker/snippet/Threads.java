//date: 2023-02-09T16:51:46Z
//url: https://api.github.com/gists/f64351ca955b504ce35431ca1aa100b3
//owner: https://api.github.com/users/evilthreads669966

public class Main {
    public static void main(String[] args) throws InterruptedException {
        final Thread t = new Thread(new MyRunnable());
        t.start();
        for (int i = 0; i < 20; i++) {
            Thread.sleep(1000);
            System.out.println("MAIN");
        }
    }
}

class MyRunnable implements Runnable{
    @Override
    public void run() {
        try {
            for(int i = 1000; i < 10000; i += 1000){
                Thread.sleep(i);
                System.out.println("Working");
            }
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}