//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.ui.activity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import com.google.android.gms.auth.api.signin.GoogleSignIn;
import com.google.android.gms.auth.api.signin.GoogleSignInAccount;
import com.google.android.gms.common.SignInButton;
import com.google.android.gms.common.api.ApiException;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.AuthCredential;
import com.google.firebase.auth.GoogleAuthProvider;
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.base.BaseActivity;
import com.suncode.relicbatik.model.User;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class LoginActivity extends BaseActivity implements View.OnClickListener {

    private static final String TAG = "LoginActivityTag";

    private SignInButton btnLogin;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);

        init();
        setData();
    }

    private void init() {
        btnLogin = findViewById(R.id.button_login_sign);
    }

    private void setData() {
        btnLogin.setOnClickListener(this);
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.button_login_sign:
                login();
                break;
        }
    }

    private void login() {
        Intent intent = mGoogleSignInClient.getSignInIntent();

        //show loading
        showLoading(true);

        //execute result
        mActivityLauncher.launch(intent, result -> {
            if (result.getResultCode() != RESULT_OK) {
                toast("Failed to signin");
                Log.d(TAG, "Error login because result code not ok");
                return;
            }

            Task<GoogleSignInAccount> task = GoogleSignIn.getSignedInAccountFromIntent(result.getData());

            try {
                GoogleSignInAccount account = task.getResult(ApiException.class);

                if (account == null) {
                    showLoading(false);
                    toast("Failed to signin");
                    Log.d(TAG, "Error login because account null");
                } else {
                    String email = account.getEmail();
                    String name = account.getDisplayName();

                    createUserInDatabase(name, email, account);
                }
            } catch (ApiException e) {
                showLoading(false);
                toast("Failed to signin");
                Log.d(TAG, "Error login because " + e.getMessage());
            }
        });
    }

    //function for creating user in database via api
    private void createUserInDatabase(String name, String email, GoogleSignInAccount account) {
        Call<User> call = mApiService.createOrUpdateClient(name, email);
        call.enqueue(new Callback<User>() {
            @Override
            public void onResponse(Call<User> call, Response<User> response) {
                if (response.body() == null) {
                    showLoading(false);
                    toast("Failed to signin");
                    Log.d(TAG, "onResponse: response null");
                    return;
                }

                if (!response.isSuccessful() || response.code() != 200) {
                    showLoading(false);
                    toast("Failed to signin");
                    Log.d(TAG, "onResponse: response not 200");
                    return;
                }

                signInCredential(account, response.body());
            }

            @Override
            public void onFailure(Call<User> call, Throwable t) {

            }
        });
    }

    //signin to google firebase
    private void signInCredential(GoogleSignInAccount account, User user) {
        showLoading(true);
        //signInWithCredential
        AuthCredential credential = "**********"
        mAuth.signInWithCredential(credential)
                .addOnCompleteListener(task -> {
                    if (!task.isSuccessful()) {
                        showLoading(false);
                        toast("Failed to signin");
                        Log.d(TAG, "Error login because task not successful");
                        return;
                    }

                    session.setUser(user);
                    showLoading(false);
                    finish();
                    toast("Signin Successfully");
                })
                .addOnFailureListener(e -> {
                    showLoading(false);
                    toast("Failed to signin");
                    Log.d(TAG, "Error login because " + e.getMessage());
                });
    }
} " + e.getMessage());
                });
    }
}