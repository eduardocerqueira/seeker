//date: 2024-02-06T16:45:52Z
//url: https://api.github.com/gists/1d93b3af6501b81c9b50b25cd2c73ae6
//owner: https://api.github.com/users/pratham1261

package com.example.expr6_2;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.FrameLayout;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        FrameLayout f1=findViewById(R.id.f1);
        TextView t1=findViewById(R.id.t1);
        displaydt(t1);
    }
    private void displaydt(TextView t1)
    {
        int intval=28;
        float floatval=15.25f;
        String Stringval="prathamesh sardesai";
        boolean boolval=true;
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("Integer: ").append(intval).append("\n");
        stringBuilder.append("Float: ").append(floatval).append("\n");
        stringBuilder.append("String: ").append(Stringval).append("\n");
        stringBuilder.append("Boolean: ").append(boolval);
        t1.setText(stringBuilder.toString());
    }
}

