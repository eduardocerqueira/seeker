//date: 2022-12-26T17:08:39Z
//url: https://api.github.com/gists/fdb83369d62d1da4084b57ada8a503b6
//owner: https://api.github.com/users/premswaroopmusti


package com.example.loginandregistrationform;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

public class RegisterActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        TextView btn = findViewById(R.id.alreadyhaveaccount);
        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(RegisterActivity.this, LoginActivity.class));
            }
        });
    }
}