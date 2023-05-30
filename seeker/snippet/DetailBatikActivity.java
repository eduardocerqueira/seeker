//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.ui.activity;

import android.content.Intent;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;
import androidx.recyclerview.widget.StaggeredGridLayoutManager;

import com.google.android.material.chip.Chip;
import com.google.android.material.chip.ChipGroup;
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.adapter.ImageBatikAdapter;
import com.suncode.relicbatik.base.BaseActivity;
import com.suncode.relicbatik.base.Constant;
import com.suncode.relicbatik.model.ApiBaseResponse;
import com.suncode.relicbatik.model.Batik;
import com.suncode.relicbatik.model.ImageBatik;

import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class DetailBatikActivity extends BaseActivity {

    private int mBatikId;

    private TextView tvName;
    private TextView tvCharacteristic;
    private TextView tvPhilosophy;
    private ChipGroup cgOrigin;
    private RecyclerView rvImage;
    private View emptyView;
    private ImageBatikAdapter mImageBatikAdapter;
    private boolean hasLike = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detail_batik);

        init();
        setData();
    }

    private void init() {
        //get data from intent
        mBatikId = getIntent().getIntExtra(Constant.INTENT_BATIK_ID, 0);

        tvName = findViewById(R.id.textview_detail_batik_name);
        tvCharacteristic = findViewById(R.id.textview_detail_batik_characteristic);
        tvPhilosophy = findViewById(R.id.textview_detail_batik_philosophy);
        cgOrigin = findViewById(R.id.chipgroup_detail_batik_origin);
        rvImage = findViewById(R.id.recycle_image_batik);
        emptyView = findViewById(R.id.emptyview_image_batik);

        mImageBatikAdapter = new ImageBatikAdapter(this, emptyView, url -> {
            Intent intent = new Intent(DetailBatikActivity.this, ImagePreviewActivity.class);
            intent.putExtra(Constant.INTENT_IMAGE_URL, url);
            startActivity(intent);
        });
    }

    private void setData() {
        //set detail batik
        getBatikDetail();

        //set rv image batik
        StaggeredGridLayoutManager layoutManager = new StaggeredGridLayoutManager(4, RecyclerView.VERTICAL);
        rvImage.setLayoutManager(layoutManager);
        rvImage.setNestedScrollingEnabled(false);
        rvImage.setAdapter(mImageBatikAdapter);

        getImageBatik();
    }

    //function for getting image batik
    private void getImageBatik() {
        Call<List<ImageBatik>> call = mApiService.getImageBatik(mBatikId);
        call.enqueue(new Callback<List<ImageBatik>>() {
            @Override
            public void onResponse(Call<List<ImageBatik>> call, Response<List<ImageBatik>> response) {
                if (response.body() == null || response.code() != 200) {
                    toast("Failed to show image");
                    return;
                }

                mImageBatikAdapter.addItems(response.body());
            }

            @Override
            public void onFailure(Call<List<ImageBatik>> call, Throwable t) {

            }
        });
    }

    //function for getting batik detail data
    private void getBatikDetail() {
        Call<List<Batik>> call = mApiService.getBatiks(mBatikId, null, null);
        call.enqueue(new Callback<List<Batik>>() {
            @Override
            public void onResponse(Call<List<Batik>> call, Response<List<Batik>> response) {
                if (response.body() == null || response.code() != 200) {
                    toast("Failed to show data");
                    return;
                }

                Batik batik = response.body().get(0);

                //set data
                tvName.setText(batik.getName());
                tvCharacteristic.setText(batik.getCharacteristic());
                tvPhilosophy.setText(batik.getPhilosophy());

                //set origin
                for (String origin : batik.getSplitOrigin()) {
                    Chip chip = new Chip(DetailBatikActivity.this);
                    chip.setText(origin);
                    chip.setChipBackgroundColorResource(R.color.purple_500);
                    chip.setTextColor(getColor(R.color.white));

                    cgOrigin.addView(chip);
                }

            }

            @Override
            public void onFailure(Call<List<Batik>> call, Throwable t) {

            }
        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.detail_batik_menu, menu);
        return true;
    }

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        super.onPrepareOptionsMenu(menu);

        //fungsi untuk mengecheck apakah user ini pernah menyukai batik tersebut
        if (session.getUser() != null) {
            Call<List<Batik>> call = mApiService.getLikeBatik(mBatikId, session.getUser().getId());
            call.enqueue(new Callback<List<Batik>>() {
                @Override
                public void onResponse(Call<List<Batik>> call, Response<List<Batik>> response) {
                    if (response.code() != 200 || response.body() == null) {
                        return;
                    }

                    if (response.body().size() > 0) {
                        //masukkan variable global untuk mempermudah like
                        hasLike = true;
                        menu.findItem(R.id.action_like).setIcon(
                                ContextCompat.getDrawable(getApplicationContext(), R.drawable.ic_like_detail_batik_menu)
                        );
                    } else {
                        menu.findItem(R.id.action_like).setIcon(
                                ContextCompat.getDrawable(getApplicationContext(), R.drawable.ic_like_border_detail_batik_menu)
                        );
                    }
                }

                @Override
                public void onFailure(Call<List<Batik>> call, Throwable t) {

                }
            });
        } else {
            menu.findItem(R.id.action_like).setIcon(
                    ContextCompat.getDrawable(this, R.drawable.ic_like_border_detail_batik_menu)
            );
        }

        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_like:
                if (session.getUser() == null) {
                    startActivity(new Intent(getApplicationContext(), LoginActivity.class));
                } else {
                    toogleLike(item);
                }
                break;
            case R.id.action_store:
                Intent intent = new Intent(DetailBatikActivity.this, StoreActivity.class);
                intent.putExtra(Constant.INTENT_BATIK_ID, mBatikId);
                startActivity(intent);
                break;
        }
        return super.onOptionsItemSelected(item);
    }

    private void toogleLike(MenuItem item) {
        showLoading(true);

        Call<ApiBaseResponse> call;

        //validate call
        if (hasLike) {
            hasLike = false;
            //change icon
            item.setIcon(ContextCompat.getDrawable(getApplicationContext(), R.drawable.ic_like_border_detail_batik_menu));

            //execute api
            call = mApiService.deleteMappingUserBatikLike(mBatikId, session.getUser().getId());
        } else {
            hasLike = true;
            //change icon
            item.setIcon(ContextCompat.getDrawable(getApplicationContext(), R.drawable.ic_like_detail_batik_menu));

            //execute api
            call = mApiService.insertIntoMappingUserBatikLike(mBatikId, session.getUser().getId());
        }

        call.enqueue(new Callback<ApiBaseResponse>() {
            @Override
            public void onResponse(Call<ApiBaseResponse> call, Response<ApiBaseResponse> response) {
                showLoading(false);
            }

            @Override
            public void onFailure(Call<ApiBaseResponse> call, Throwable t) {
                showLoading(false);
            }
        });
    }
}