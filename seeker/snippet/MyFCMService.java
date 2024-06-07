//date: 2024-06-07T16:52:30Z
//url: https://api.github.com/gists/2843257bb5f848d82d1b59406b3610be
//owner: https://api.github.com/users/AyushPorwal10

package com.example.let;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.core.app.NotificationCompat;

import com.google.firebase.messaging.FirebaseMessagingService;
import com.google.firebase.messaging.RemoteMessage;

import java.util.Map;

public class MyFCMService extends FirebaseMessagingService {
    NotificationManager mNotificationManager;
    @Override

    public void onNewToken(@NonNull String token){
        super.onNewToken(token);
        updateToken(token);
    }
    @Override
    public void onMessageReceived(@NonNull RemoteMessage message) {
        super.onMessageReceived(message);
        Log.d("Data aa gyaa",message.getFrom());

        Map<String , String> data = message.getData();
        NotificationCompat.Builder builder = new NotificationCompat.Builder(this,"channel_id");
        Intent resultIntent = new Intent(this,Chat_Activity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(this,1,resultIntent,PendingIntent.FLAG_IMMUTABLE);

        builder.setContentTitle(data.get("title"));
        builder.setContentText(data.get("body"));
        builder.setStyle(new NotificationCompat.BigTextStyle().bigText(data.get("body")));
        builder.setAutoCancel(true);
        builder.setPriority(Notification.PRIORITY_DEFAULT);
        builder.setSmallIcon(R.drawable.logo);
        mNotificationManager = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel("channel_id","channel_name",NotificationManager.IMPORTANCE_DEFAULT);
            mNotificationManager.createNotificationChannel(channel);
            builder.setChannelId("channel_id");
        }
        mNotificationManager.notify(100,builder.build());

    }
    public void updateToken(String tokenToUpdate){

    }
}
