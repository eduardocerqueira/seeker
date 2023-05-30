//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.ui.activity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.widget.PopupMenu;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.adapter.BatikAdapter;
import com.suncode.relicbatik.adapter.TopBatikLikeAdapter;
import com.suncode.relicbatik.base.BaseActivity;
import com.suncode.relicbatik.base.Constant;
import com.suncode.relicbatik.model.Batik;
import com.suncode.relicbatik.model.TopBatikLike;

import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends BaseActivity implements View.OnClickListener {

    private static final String TAG = "MainActivityTag";

    private FloatingActionButton fabMain;
    private ImageView ivTopMenuButton;
    private View emptyViewTopBatikLike, emptyViewBatik;
    private RecyclerView rvTopBatikLike, rvBatik;
    private TopBatikLikeAdapter topBatikLikeAdapter;
    private BatikAdapter batikAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        init();
        setData();
    }

    //function for initialize component
    private void init() {
        fabMain = findViewById(R.id.fab_main_scan);
        ivTopMenuButton = findViewById(R.id.imageView_main_more_menu);
        emptyViewTopBatikLike = findViewById(R.id.emptyview_main_top_like_batik);
        emptyViewBatik = findViewById(R.id.emptyview_main_batik);
        rvTopBatikLike = findViewById(R.id.recycle_top_like_batik);
        rvBatik = findViewById(R.id.recycle_batik);

        topBatikLikeAdapter = new TopBatikLikeAdapter(this, emptyViewTopBatikLike, (position, topBatikLike) -> {
            //clickable top batik like
            Intent intent = new Intent(MainActivity.this, DetailBatikActivity.class);
            intent.putExtra(Constant.INTENT_BATIK_ID, topBatikLike.getBatikId());
            startActivity(intent);
        });

        batikAdapter = new BatikAdapter(this, emptyViewBatik, batik -> {
            //clickable batik
            Intent intent = new Intent(MainActivity.this, DetailBatikActivity.class);
            intent.putExtra(Constant.INTENT_BATIK_ID, batik.getId());
            startActivity(intent);
        });
    }

    //function for setting component
    private void setData() {
        fabMain.setOnClickListener(this);
        ivTopMenuButton.setOnClickListener(this);

        //set data top batik like
        topBatikLike();

        //set data batik
        batik();
    }

    //function for setup data batik
    private void batik() {
        LinearLayoutManager layoutManager = new LinearLayoutManager(this, RecyclerView.VERTICAL, false) {
            @Override
            public boolean canScrollVertically() {
                return false;
            }
        };
        rvBatik.setLayoutManager(layoutManager);
        rvBatik.setNestedScrollingEnabled(false);
        rvBatik.setAdapter(batikAdapter);

        //get data
        getBatik();
    }

    //function for getting data batik
    private void getBatik() {
        Call<List<Batik>> call = mApiService.getDashboardBatik();
        call.enqueue(new Callback<List<Batik>>() {
            @Override
            public void onResponse(Call<List<Batik>> call, Response<List<Batik>> response) {
                if (response.body() == null || response.code() != 200) {
                    Log.d(TAG, "onResponse: " + response.code());
                    return;
                }

                //set to adapter
                batikAdapter.addItems(response.body());
            }

            @Override
            public void onFailure(Call<List<Batik>> call, Throwable t) {

            }
        });
    }

    //function for setup data top batik like
    private void topBatikLike() {
        LinearLayoutManager layoutManager = new LinearLayoutManager(this, RecyclerView.VERTICAL, false) {
            @Override
            public boolean canScrollVertically() {
                return false;
            }
        };
        rvTopBatikLike.setLayoutManager(layoutManager);
        rvBatik.setNestedScrollingEnabled(false);
        rvTopBatikLike.setAdapter(topBatikLikeAdapter);

        //get data
        getTopBatikLike();
    }

    //function for getting data top batik like
    private void getTopBatikLike() {
        Call<List<TopBatikLike>> call = mApiService.getTopBatikLike();
        call.enqueue(new Callback<List<TopBatikLike>>() {
            @Override
            public void onResponse(Call<List<TopBatikLike>> call, Response<List<TopBatikLike>> response) {
                if (response.body() == null || response.code() != 200) {
                    Log.d(TAG, "onResponse: " + response.code());
                    return;
                }

                //set to adapter
                topBatikLikeAdapter.addItem(response.body());
            }

            @Override
            public void onFailure(Call<List<TopBatikLike>> call, Throwable t) {

            }
        });
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.fab_main_scan:
                //validate permission
                if (!isCameraPermissionGranted() || !isReadStroragePermissionGranted() || !isAudioPermissionGranted()) {
                    openDialogPermission();
                } else {
                    startActivity(new Intent(MainActivity.this, CameraActivity.class));
                }
                break;
            case R.id.imageView_main_more_menu:
                openMenu();
                break;
        }
    }

    //function for open menu
    private void openMenu() {
        PopupMenu popupMenu = new PopupMenu(this, ivTopMenuButton);
        popupMenu.getMenuInflater().inflate(R.menu.main_top_menu, popupMenu.getMenu());

        //clickable
        popupMenu.setOnMenuItemClickListener(item -> {
            switch (item.getItemId()) {
                case R.id.action_profile:
                    popupMenu.dismiss();
                    startActivity(new Intent(MainActivity.this, ProfileActivity.class));
                    break;
            }
            return false;
        });

        //show popup
        popupMenu.show();
    }

    //function for open dialog permission camera
    private void openDialogPermission() {
        AlertDialog.Builder builder = dialogMessage(
                MainActivity.this,
                "Camera and File Storage Access Required",
                "Camera and media storage access permission is required to access this feature"
        );

        builder.setPositiveButton("Allow", (dialogInterface, i) -> {
            dialogInterface.dismiss();

            //get permission
            getPermissionCameraAudioAndReadStorage();
            toast("Tap scan button again");
        });

        builder.setNegativeButton("Cancel", ((dialogInterface, i) -> dialogInterface.dismiss()));

        //show dialog
        builder.create().show();
    }
}