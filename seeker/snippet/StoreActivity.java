//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.ui.activity;

import androidx.appcompat.app.AlertDialog;
import androidx.core.content.ContextCompat;
import androidx.core.location.LocationManagerCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.location.Location;
import android.location.LocationManager;
import android.os.Bundle;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.Button;

import com.suncode.relicbatik.R;
import com.suncode.relicbatik.adapter.StoreAdapter;
import com.suncode.relicbatik.base.BaseActivity;
import com.suncode.relicbatik.base.Constant;
import com.suncode.relicbatik.model.Store;

import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class StoreActivity extends BaseActivity implements View.OnClickListener {

    private static final String TAG = "StoreActivityTag";

    private int mBatikId;

    private StoreAdapter mStoreAdapter;

    private View vLocationPermission;
    private View vGpsDisable;
    private View vStoreData;
    private View mEmptyView;
    private Button btnLocationPermission;
    private Button btnAvailableStore;
    private Button btnReloadData;
    private RecyclerView rvStore;
    private SwipeRefreshLayout swipeStore;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_store);

        init();
        setData();
    }

    //initialize component
    private void init() {
        //get batik id from intent
        mBatikId = getIntent().getIntExtra(Constant.INTENT_BATIK_ID, 0);

        //component
        vLocationPermission = findViewById(R.id.view_store_location_permission);
        vGpsDisable = findViewById(R.id.view_store_gps_disable);
        vStoreData = findViewById(R.id.view_store_data);
        mEmptyView = findViewById(R.id.emptyview_store);
        btnLocationPermission = findViewById(R.id.button_store_allow_location_permission);
        btnReloadData = findViewById(R.id.button_store_reload_data);
        btnAvailableStore = findViewById(R.id.button_emptyview_store_available_store);
        rvStore = findViewById(R.id.recycle_store);
        swipeStore = findViewById(R.id.swiperefresh_store);

        //adapter
        mStoreAdapter = new StoreAdapter(this, mEmptyView, store -> {
            //clickable store
            Intent intent = new Intent(StoreActivity.this, DetailStoreActivity.class);
            intent.putExtra(Constant.INTENT_STORE, store);
            intent.putExtra(Constant.INTENT_BATIK_ID, mBatikId);
            startActivity(intent);
        });
    }

    //setup component data
    private void setData() {
        //recycleview setup
        LinearLayoutManager layoutManager = new LinearLayoutManager(this, RecyclerView.VERTICAL, false);
        rvStore.setLayoutManager(layoutManager);
        rvStore.setAdapter(mStoreAdapter);

        //button clicked
        btnLocationPermission.setOnClickListener(this);
        btnReloadData.setOnClickListener(this);
        btnAvailableStore.setOnClickListener(this);

        //validate permission location
        if (!isLocationPermissionGranted()) {
            showView(true, false, false);
            return;
        }

        //validate location enable
        if (!isLocationEnabled()) {
            showView(false, true, false);
            return;
        }

        //get store data
        showStore(false);

        //swipe refreshing effect
        swipeStore.setOnRefreshListener(() -> {
            //validate location enable
            if (!isLocationEnabled()) {
                showView(false, true, false);
                return;
            }

            showStore(false);
        });
    }

    //function for showing store
    private void showStore(boolean isAnotherLocation) {
        //validate lotitude longitude
        Pair<Double, Double> currentLocation = getCurrentLocation();

        if (currentLocation.first == null || currentLocation.second == null) {
            showView(false, true, false);
            return;
        }

        //show loading
        swipeStore.setRefreshing(true);

        //validate function another location for getting another data
        Call<List<Store>> call;

        if (isAnotherLocation) {
            call = mApiService.getStoreWithBatikAndCurrentLocationAnotherData(
                    currentLocation.first,
                    currentLocation.second,
                    mBatikId,
                    ""
            );
        } else {
            call = mApiService.getStoreWithBatikAndCurrentLocation(
                    currentLocation.first,
                    currentLocation.second,
                    mBatikId
            );
        }

        call.enqueue(new Callback<List<Store>>() {
            @Override
            public void onResponse(Call<List<Store>> call, Response<List<Store>> response) {
                swipeStore.setRefreshing(false);

                if (response.body() == null || response.code() != 200) {
                    toast("Failed to load data");
                    return;
                }

                mStoreAdapter.addItems(response.body());
            }

            @Override
            public void onFailure(Call<List<Store>> call, Throwable t) {
                swipeStore.setRefreshing(false);
                Log.d(TAG, "onFailure: " + t.getMessage());
            }
        });
    }

    //function for show group view
    private void showView(boolean locationPermission, boolean gpsDiable, boolean storeData) {
        vLocationPermission.setVisibility((locationPermission) ? View.VISIBLE : View.GONE);
        vGpsDisable.setVisibility((gpsDiable) ? View.VISIBLE : View.GONE);
        vStoreData.setVisibility((storeData) ? View.VISIBLE : View.GONE);
    }

    //function for getting status location
    private boolean isLocationEnabled() {
        LocationManager locationManager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);
        return LocationManagerCompat.isLocationEnabled(locationManager);
    }

    //function for getting current location
    private Pair<Double, Double> getCurrentLocation() {
        return new Pair<>(getLastKnownLocation().getLatitude(), getLastKnownLocation().getLongitude());
    }

    //function for getting last know location
    private Location getLastKnownLocation() {
        Location l = null;
        LocationManager mLocationManager = (LocationManager) getApplicationContext().getSystemService(LOCATION_SERVICE);
        List<String> providers = mLocationManager.getProviders(true);
        Location bestLocation = null;
        for (String provider : providers) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
                l = mLocationManager.getLastKnownLocation(provider);
            }
            if (l == null) {
                continue;
            }
            if (bestLocation == null || l.getAccuracy() < bestLocation.getAccuracy()) {
                bestLocation = l;
            }
        }
        return bestLocation;
    }


    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.button_store_allow_location_permission:
                getPermissionLocation();
                break;
            case R.id.button_store_reload_data:
                //validate location enable
                if (!isLocationEnabled()) {
                    showView(false, true, false);
                } else {
                    showView(false, false, true);
                }
                break;
            case R.id.button_emptyview_store_available_store:
                showStore(true);
                break;
        }
    }
}