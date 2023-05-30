//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.ui.activity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.suncode.relicbatik.R;
import com.suncode.relicbatik.adapter.BatikAdapter;
import com.suncode.relicbatik.base.BaseActivity;
import com.suncode.relicbatik.base.Constant;
import com.suncode.relicbatik.model.Batik;

import java.util.ArrayList;
import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class FavoriteActivity extends BaseActivity {

    private RecyclerView rvFavorit;
    private View mEmptyView;
    private SwipeRefreshLayout swipeFavorite;

    private BatikAdapter mAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_favorite);

        init();
        setData();
    }

    private void init() {
        rvFavorit = findViewById(R.id.recycle_favorite);
        mEmptyView = findViewById(R.id.emptyview_favorite);
        swipeFavorite = findViewById(R.id.swiperefresh_favorite);

        mAdapter = new BatikAdapter(this, mEmptyView, batik -> {
            //clickable batik
            Intent intent = new Intent(FavoriteActivity.this, DetailBatikActivity.class);
            intent.putExtra(Constant.INTENT_BATIK_ID, batik.getId());
            startActivity(intent);
        });
    }

    private void setData() {
        LinearLayoutManager layoutManager = new LinearLayoutManager(this, RecyclerView.VERTICAL, false);
        rvFavorit.setLayoutManager(layoutManager);
        rvFavorit.setAdapter(mAdapter);

        //set refresh layout favorite
        swipeFavorite.setOnRefreshListener(() -> {
            mAdapter.addItems(new ArrayList<>());
            getFavoriteBatik();
        });

        //get data favorite
        getFavoriteBatik();
    }

    //function for getting batik favorite
    private void getFavoriteBatik() {
        //active refreshing
        swipeFavorite.setRefreshing(true);
//        swipeFavorite.setEnabled(true);

        Call<List<Batik>> call = mApiService.getLikeBatik(null, session.getUser().getId());
        call.enqueue(new Callback<List<Batik>>() {
            @Override
            public void onResponse(Call<List<Batik>> call, Response<List<Batik>> response) {
                //deactive refreshing
                swipeFavorite.setRefreshing(false);
//                swipeFavorite.setEnabled(false);

                if (response.body() == null || response.code() != 200) {
                    toast("Failed to load data");
                    return;
                }

                mAdapter.addItems(response.body());
            }

            @Override
            public void onFailure(Call<List<Batik>> call, Throwable t) {
                //deactive refreshing
                swipeFavorite.setRefreshing(false);
            }
        });
    }
}