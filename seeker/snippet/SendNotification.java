//date: 2024-06-07T16:52:30Z
//url: https://api.github.com/gists/2843257bb5f848d82d1b59406b3610be
//owner: https://api.github.com/users/AyushPorwal10

package com.example.let;

import android.content.Context;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.NonNull;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.example.let.utils.AndroidUtil;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;

public class SendNotification {
    private final String userFCMToken;
    private final String title;
    private  final String body ;
    private final Context context;
    private final String postUrl = "https://fcm.googleapis.com/v1/projects/fir-learn-90292/messages:send";

    public SendNotification(String userFCMToken, String title, String body, Context context) {
        Log.d("Constructor recived ",userFCMToken +"ERROR");
        this.userFCMToken = "**********"
        this.title = title;
        this.body = body;
        this.context = context;
    }
    public void sendNotification(){
        RequestQueue requestQueue = Volley.newRequestQueue(context);
        JSONObject mainObj = new JSONObject();
        try{
            JSONObject messageObject = new JSONObject();
            messageObject.put("token",userFCMToken);
            JSONObject notificationObject = new JSONObject();
            notificationObject.put("title",title);
            notificationObject.put("body",body);

            JSONObject dataObject = new JSONObject();
            dataObject.put("title", title);
            dataObject.put("body", body);


            // Attach the notification to the message object
            messageObject.put("notification", notificationObject);
            messageObject.put("data",dataObject);

            // Set the message object as the root of the JSON payload
            mainObj.put("message", messageObject);

            Log.d("request_body", mainObj.toString());
            JsonObjectRequest request = new JsonObjectRequest(Request.Method.POST,postUrl,mainObj,response ->
                    Log.d("Notification aa gyaa", "Notification sent successfully"),

            volleyError -> {
                Log.e("Volley error", "Failed to send notification: " + volleyError.toString());
                Log.e("Volley error", "Error response: " + new String(volleyError.networkResponse.data));
                AndroidUtil.showToast(context, "Failed to send notification");
            }
            ){
                @NonNull
                @Override
                public Map<String , String > getHeaders(){
                    AccessToken accessToken = "**********"
                    String accessKey = "**********"

                    Map<String , String > header = new HashMap<>();
                    header.put("Context-type","application/json");

                    Log.d("accesskey",accessKey);
                    header.put("Authorization","Bearer "+accessKey);
                    Log.d("Header","Header is returned");
                    return header;
                }
            };
            requestQueue.add(request);
        }
        catch (JSONException e){
            AndroidUtil.showToast(context,e.getMessage()+"");
        }

    }


}
);
        }

    }


}
