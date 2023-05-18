//date: 2023-05-18T17:02:30Z
//url: https://api.github.com/gists/ec15447e119405676320b5fa3be0a4d5
//owner: https://api.github.com/users/pranavpa8788

package com.example.androidtest;

import android.os.Bundle;
import androidx.fragment.app.FragmentActivity;
import androidx.viewpager2.widget.ViewPager2;

public class MainActivity extends FragmentActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        ViewPager2 view_pager = findViewById(R.id.view_pager);
        view_pager.setAdapter(new CustomAdapter(this));
    }

}