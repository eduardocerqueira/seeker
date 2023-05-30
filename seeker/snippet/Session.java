//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.base;

import android.content.Context;
import android.content.SharedPreferences;

import com.google.gson.Gson;
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.model.User;

public class Session {
    private final SharedPreferences preferences;
    private final SharedPreferences.Editor editor;
    private final Context context;
    private final Gson gson = new Gson();

    public Session(Context context) {
        this.context = context;
        preferences = context.getSharedPreferences(context.getString(R.string.app_name), Context.MODE_PRIVATE);
        editor = preferences.edit();
    }

    public void setIntroStatus(boolean status) {
        editor.putBoolean(Constant.SESSION_INTRO, status);
        editor.commit();
    }

    public boolean getIntroStatus() {
        return preferences.getBoolean(Constant.SESSION_INTRO, false);
    }

    public void setUser(User user) {
        if (user == null) {
            editor.putString(Constant.SESSION_USER, null);
            editor.commit();
        } else {
            String userString = gson.toJson(user);

            editor.putString(Constant.SESSION_USER, userString);
            editor.commit();
        }
    }

    public User getUser() {
        String user = preferences.getString(Constant.SESSION_USER, null);
        return (user == null) ? null : gson.fromJson(user, User.class);
    }

    public void setCameraInfoStatus(boolean status) {
        editor.putBoolean(Constant.SESSION_INFO_CAMERA, status);
        editor.commit();
    }

    public boolean getCameraInfoStatus() {
        return preferences.getBoolean(Constant.SESSION_INFO_CAMERA, false);
    }
}
