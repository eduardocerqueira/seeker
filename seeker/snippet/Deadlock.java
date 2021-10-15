//date: 2021-10-15T16:55:00Z
//url: https://api.github.com/gists/fc165856ab7d0c4bb0d407f79613d3ca
//owner: https://api.github.com/users/mitulvaghamshi

public class Deadlock {
	public static void main(String[] args) throws Exception {
		Object resourceA = new Object();
		Object resourceB = new Object();
		Thread threadLockingResourceAFirst = new Thread(new DeadlockRunnable(resourceA, resourceB));
		Thread threadLockingResourceBFirst = new Thread(new DeadlockRunnable(resourceB, resourceA));
		threadLockingResourceAFirst.start();
		Thread.sleep(500);
		threadLockingResourceBFirst.start();
	}
	private static class DeadlockRunnable implements Runnable {
		private final Object firstResource;
		private final Object secondResource;
		public DeadlockRunnable(Object firstResource, Object secondResource) {
			this.firstResource=firstResource;
			this.secondResource=secondResource;
		}
		@Override
		public void run() {
			try {
				synchronized(firstResource) {
					printLockedResource(firstResource);
					Thread.sleep(500);
					synchronized(secondResource) {
						printLockedResource(secondResource);
					}
				}
			} catch(InterruptedException e) {
				System.out.println("Exception occurred: "+e);
			}
		}
		private static void printLockedResource(Object resource) {
			System.out.println(Thread.currentThread().getName()+": locked resource -> "+resource);
		}
	}
}
