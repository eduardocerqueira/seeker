//date: 2023-06-20T17:08:02Z
//url: https://api.github.com/gists/7d3d29d208ee3a4ec038bf8ef89c1124
//owner: https://api.github.com/users/adistwn

package varunon9.me.dynamicwallpaper;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

import androidx.work.OneTimeWorkRequest;
import androidx.work.WorkManager;

public class MyReceiver extends BroadcastReceiver {
    private String TAG = "MyReceiver";

    @Override
    public void onReceive(Context context, Intent intent) {
        Log.d(TAG, "onReceive called");

        // We are starting MyService via a worker and not directly because since Android 7
        // (but officially since Lollipop!), any process called by a BroadcastReceiver
        // (only manifest-declared receiver) is run at low priority and hence eventually
        // killed by Android.
        WorkManager workManager = WorkManager.getInstance(context);
        OneTimeWorkRequest startServiceRequest = new OneTimeWorkRequest.Builder(MyWorker.class)
                .build();
        workManager.enqueue(startServiceRequest);
    }
}
