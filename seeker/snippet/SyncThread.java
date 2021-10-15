//date: 2021-10-15T16:55:49Z
//url: https://api.github.com/gists/e25d93e7c8916f6d1f9fe1f58fc19f6f
//owner: https://api.github.com/users/mitulvaghamshi

public class SyncThread {
	public static void main(String[] argc) {
		Printer printer = new Printer();
		// new runme(p,16);
		// new runme(p,10);
		// new runme(p,8);
		new Thread(new RunMe(printer, 16)).start();
		new Thread(new RunMe(printer, 10)).start();
		new Thread(new RunMe(printer, 8)).start();
		new Thread(new RunMe(printer, 4)).start();
	}
}

class RunMe implements Runnable {
	int n;
	Printer printer;

	RunMe(Printer p, int n) {
		this.printer = p;
		this.n = n;
		// new Thread(this).start();
	}

	@Override
	public void run() {
		printer.printNums(this.n);
	}
}

class Printer {
	/* synchronized */ void printNums(int n) { // synchronized method
		synchronized (this) { // synchronized block
			System.out.print("\nSTART");
			for (int j = n; j > 0; j--) {
				if (n / 2 == j)
					try {
						Thread.sleep(100);
					} catch (InterruptedException e) {
					}
				System.out.print(" " + j);
			}
			System.out.println(" END");
		}
	}
}
