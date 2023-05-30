//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.ui.activity;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.os.Bundle;
import android.view.MenuItem;
import android.widget.ImageView;

import com.bumptech.glide.Glide;
import com.bumptech.glide.load.engine.DiskCacheStrategy;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.adapter.AccountAdapter;
import com.suncode.relicbatik.base.BaseActivity;
import com.suncode.relicbatik.base.BaseFunction;
import com.suncode.relicbatik.base.Constant;
import com.suncode.relicbatik.model.Menu;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class AccountActivity extends BaseActivity {

    private ImageView ivPhoto;
    private RecyclerView rvAccount;
    private AccountAdapter mAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_account);

        init();
        setData();
    }

    private void init() {
        ivPhoto = findViewById(R.id.imageView_account_picture);
        rvAccount = findViewById(R.id.recycle_account);
        mAdapter = new AccountAdapter(this, menuList());
    }

    private void setData() {
        LinearLayoutManager layoutManager = new LinearLayoutManager(this, RecyclerView.VERTICAL, false) {
            @Override
            public boolean canScrollVertically() {
                return false;
            }
        };

        //get data user
        FirebaseUser user = mAuth.getCurrentUser();

        //set photo
        if (user.getPhotoUrl() != null) {
            Glide.with(this)
                    .load(user.getPhotoUrl())
                    .diskCacheStrategy(DiskCacheStrategy.DATA)
                    .into(ivPhoto);
        }

        rvAccount.setLayoutManager(layoutManager);
        rvAccount.setAdapter(mAdapter);
    }

    private List<Menu> menuList() {
        //get data user
        FirebaseUser user = mAuth.getCurrentUser();

        List<Menu> menus = new ArrayList<>();

        menus.add(new Menu(
                "Name",
                user.getDisplayName(),
                ContextCompat.getDrawable(this, R.drawable.ic_person)
        ));

        menus.add(new Menu(
                "Email",
                user.getEmail(),
                ContextCompat.getDrawable(this, R.drawable.ic_email)
        ));

        if (user.getMetadata() == null) {
            menus.add(new Menu(
                    "Created At",
                    "-",
                    ContextCompat.getDrawable(this, R.drawable.ic_calendar)
            ));

            menus.add(new Menu(
                    "Last Signin At",
                    "-",
                    ContextCompat.getDrawable(this, R.drawable.ic_calendar)
            ));
        } else {
            menus.add(new Menu(
                    "Created At",
                    BaseFunction.millisecondsToDate(Constant.DATE_TIME_FORMAT_ACCOUNT, user.getMetadata().getCreationTimestamp()),
                    ContextCompat.getDrawable(this, R.drawable.ic_calendar)
            ));

            menus.add(new Menu(
                    "Last Signin At",
                    BaseFunction.millisecondsToDate(Constant.DATE_TIME_FORMAT_ACCOUNT, user.getMetadata().getLastSignInTimestamp()),
                    ContextCompat.getDrawable(this, R.drawable.ic_calendar)
            ));
        }


        return menus;
    }

    @Override
    public boolean onCreateOptionsMenu(android.view.Menu menu) {
        getMenuInflater().inflate(R.menu.account_menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_logout:
                logout();
                break;
        }
        return super.onOptionsItemSelected(item);
    }

    private void logout() {
        //function for logout
        AlertDialog.Builder builder = dialogMessage(
                AccountActivity.this,
                "Logout account from application",
                "Are you sure want to logout this account from application?"
        );

        //button dialog listener
        builder.setPositiveButton("Logout", (dialog, which) -> {
            FirebaseAuth.getInstance().signOut();
            session.setUser(null);
            finish();
        });

        builder.setNegativeButton("Cancel", (dialog, which) -> {
            dialog.dismiss();
        });

        //show
        AlertDialog dialog = builder.create();
        dialog.show();
    }

}