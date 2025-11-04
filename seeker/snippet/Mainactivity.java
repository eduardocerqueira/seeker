//date: 2025-11-04T17:08:54Z
//url: https://api.github.com/gists/2bb068eaa564315a00c31d2cc10b645c
//owner: https://api.github.com/users/Anjali-Prakash

package com.example.layouts; 
import android.content.Intent; 
import android.os.Bundle; 
import android.widget.RadioButton; 
import android.widget.RadioGroup; 
import androidx.activity.EdgeToEdge; 
import androidx.appcompat.app.AppCompatActivity; 
import androidx.core.graphics.Insets; 
import androidx.core.view.ViewCompat; 
import androidx.core.view.WindowInsetsCompat; 
public class MainActivity extends AppCompatActivity { 
    RadioButton rb1, rb2, rb3, rb4, rb5; 
    RadioGroup rg; 
    @Override 
    protected void onCreate(Bundle savedInstanceState) { 
        super.onCreate(savedInstanceState); 
        EdgeToEdge.enable(this); 
        setContentView(R.layout.activity_main); 
        rb1 = findViewById(R.id.ra1); 
        rb2 = findViewById(R.id.ra2); 
        rb3 = findViewById(R.id.ra3); 
        rb4 = findViewById(R.id.ra4); 
        rb5 = findViewById(R.id.ra5); 
        rg = findViewById(R.id.rgrp); 
        rg.setOnCheckedChangeListener((radioGroup, i) -> { 
            if (rb1.isChecked()) { 
                startActivity(new Intent(MainActivity.this, MainActivity6.class)); 
            } 
            if (rb2.isChecked()) { 
                startActivity(new Intent(MainActivity.this, MainActivity2.class)); 
            } 
            if (rb3.isChecked()) { 
                startActivity(new Intent(MainActivity.this, MainActivity3.class)); 
            } 
            if (rb4.isChecked()) { 
                startActivity(new Intent(MainActivity.this, MainActivity4.class)); 
            } 
            if (rb5.isChecked()) { 
                startActivity(new Intent(MainActivity.this, MainActivity5.class)); 
            } 
        }); 
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> { 
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars()); 
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom); 
            return insets; 
        }); 
    } 
} 