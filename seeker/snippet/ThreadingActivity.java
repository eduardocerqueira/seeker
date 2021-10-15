//date: 2021-10-15T16:59:04Z
//url: https://api.github.com/gists/10c367d133a11c8def948984cf449f8a
//owner: https://api.github.com/users/mitulvaghamshi

package package_name;

import android.app.Activity;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.View;
import android.widget.TextView;

public class ThreadingActivity extends Activity {
	private static TextView textView;
  // private CountingThread thread;    
	private DoCountingTask task;

  @Override
  public void onCreate(Bundle savedInstanceState) {
      super.onCreate(savedInstanceState);
      setContentView(R.layout.main);
      textView = findViewById(R.id.textView1);
  }      

  public void startCounter(View view) {  
    task = new DoCountingTask().execute();        
  }

  public void stopCounter(View view) {  
    task.cancel(true);
  }    

	static Handler UIupdater = new Handler() {
		@Override
		public void handleMessage(Message msg) {              
			textView.setText(new String((byte[]) msg.obj));
		}
	};

//	public void startCounter(View view) {  
//		thread=new CountingThread();    
//		thread.start();
//	}
//
//	public void stopCounter(View view) {  
//		thread.cancel();
//	}

//	public void startCounter(View view) {  
//		new Thread(new Runnable() {
//				@Override
//				public void run() {    			
//					for(int i=0; i<=1000; i++) {
//						ThreadingActivity.UIupdater.obtainMessage(0, String.valueOf(i).getBytes()).sendToTarget();
//						try {
//							Thread.sleep(1000);
//						} catch(InterruptedException e) {}
//					}
//				}    		
//			}).start();
//	}

//	public void startCounter(View view) {  
//		new Thread(new Runnable() {
//				@Override
//				public void run() {    			
//					for(int i=0; i<=1000; i++) {
//						final int valueOfi = i; 
//						textView.post(new Runnable() {
//								public void run() {
//									textView.setText(String.valueOf(valueOfi));
//								}
//							});
//						try {
//							Thread.sleep(1000);
//						} catch(InterruptedException e) {}
//					}
//				}    		
//			}).start();
//	}       

//	public void startCounter(View view) {
//		new Thread(new Runnable() {
//				@Override
//				public void run() {
//					for(int i=0; i<=1000; i++) {
//						textView.setText(String.valueOf(i));
//						try {
//							Thread.sleep(1000);
//						} catch(InterruptedException e) {}
//					}    
//				}    		
//			}).start();
//	}

	@Override
	protected void onPause() {
		super.onPause();
		stopCounter(textView);
	}    

	private class DoCountingTask extends AsyncTask<Void, Integer, Void> {
      @Override
      protected Void doInBackground(Void... params) {         
          for(int i = 0; i<1000; i++) {
              publishProgress(i);
              try {
        Thread.sleep(1000);
      } catch(InterruptedException e) {}
              if(isCancelled()) break;
          }
          return null;
      }

      @Override
      protected void onProgressUpdate(Integer... progress) {            
        textView.setText(progress[0].toString());
      }
  }    

//	private class CountingThread extends Thread {
//		Boolean cancel=false;
//		public void run() {
//			for(int i=0; i<=1000; i++) {
//				ThreadingActivity.UIupdater.obtainMessage(0, String.valueOf(i).getBytes()).sendToTarget();
//				try {
//					Thread.sleep(1000);
//				} catch(InterruptedException e) {}
//				if(cancel) break;
//			}
//		}
//
//		public void cancel() {
//			cancel=true;    	
//		}
//	}
}
