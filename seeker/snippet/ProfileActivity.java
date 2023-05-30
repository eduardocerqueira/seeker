//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.ui.activity;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.content.Intent;
import android.os.Bundle;

import com.suncode.relicbatik.R;
import com.suncode.relicbatik.adapter.MenuAdapter;
import com.suncode.relicbatik.base.BaseActivity;
import com.suncode.relicbatik.model.Menu;

import java.util.ArrayList;
import java.util.List;

public class ProfileActivity extends BaseActivity {

    private RecyclerView rvProfile;
    private MenuAdapter mAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_profile);

        init();
        setData();
    }

    private void init() {
        rvProfile = findViewById(R.id.recycle_profile);
        mAdapter = new MenuAdapter(this, menuList(), (position, menu) -> {
            switch (position) {
                case 0:
                    if (mAuth.getCurrentUser() == null)
                        startActivity(new Intent(ProfileActivity.this, LoginActivity.class));
                    else
                        startActivity(new Intent(ProfileActivity.this, AccountActivity.class));
                    break;
                case 1:
                    if (mAuth.getCurrentUser() == null)
                        startActivity(new Intent(ProfileActivity.this, LoginActivity.class));
                    else
                        startActivity(new Intent(ProfileActivity.this, FavoriteActivity.class));
                    break;
            }
        });
    }

    private void setData() {
        LinearLayoutManager layoutManager = new LinearLayoutManager(this, RecyclerView.VERTICAL, false);
        rvProfile.setLayoutManager(layoutManager);
        rvProfile.setAdapter(mAdapter);
    }

    private List<Menu> menuList() {
        List<Menu> menus = new ArrayList<>();

        menus.add(new Menu(
                "Account Information",
                "Information related to accounts connected in the application",
                ContextCompat.getDrawable(this, R.drawable.ic_person)
        ));

        menus.add(new Menu(
                "Favorite batiks",
                "List of all batik marked favorite",
                ContextCompat.getDrawable(this, R.drawable.ic_like)
        ));

        return menus;
    }
}