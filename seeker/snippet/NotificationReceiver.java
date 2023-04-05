//date: 2023-04-05T16:53:25Z
//url: https://api.github.com/gists/6eb54f95aabffbc748e98fbb40bdd329
//owner: https://api.github.com/users/biswa-rx

package com.codinginflow.notificationsexample;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.widget.Toast;


public class NotificationReceiver extends BroadcastReceiver {

    @Override
    public void onReceive(Context context, Intent intent) {
        String message = intent.getStringExtra("toastMessage");
        Toast.makeText(context, message, Toast.LENGTH_SHORT).show();
    }
}