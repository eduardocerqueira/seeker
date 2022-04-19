//date: 2022-04-19T16:57:40Z
//url: https://api.github.com/gists/21eb3f1d49ccce811e01cece7846f370
//owner: https://api.github.com/users/PranithChowdary

package com.example.notifications;

public class Message {

    private CharSequence text,sender;

    private long timestamp;

    public Message(CharSequence text, CharSequence sender) {

        this.text = text;

        this.sender = sender;

        this.timestamp = System.currentTimeMillis();

    }

    public CharSequence getText() {

        return text;

    }

    public void setText(CharSequence text) {

        this.text = text;

    }

    public CharSequence getSender() {

        return sender;

    }

    public void setSender(CharSequence sender) {

        this.sender = sender;

    }

    public long getTimestamp() {

        return timestamp;

    }

    public void setTimestamp(long timestamp) {

        this.timestamp = timestamp;

    }

}

------------------------------------------------

Notification_Toast.java



package com.example.notifications;

import android.app.RemoteInput;

import android.content.BroadcastReceiver;

import android.content.Context;

import android.content.Intent;

import android.os.Bundle;

import android.widget.Toast;

public class NotificationReceiverToast extends BroadcastReceiver {

    @Override

    public void onReceive(Context context, Intent intent) {

        String message = intent.getStringExtra("message");

        Toast.makeText(context, ""+message, Toast.LENGTH_SHORT).show();

    }

}