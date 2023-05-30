//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.ui.activity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.ImageView;

import androidx.recyclerview.widget.DividerItemDecoration;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.bumptech.glide.request.RequestOptions;
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.adapter.DetectionAdapter;
import com.suncode.relicbatik.base.BaseActivity;
import com.suncode.relicbatik.base.Constant;
import com.suncode.relicbatik.ml.Model;
import com.suncode.relicbatik.model.Batik;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.Category;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

import de.hdodenhof.circleimageview.CircleImageView;
import jp.wasabeef.glide.transformations.BlurTransformation;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class DetectionActivity extends BaseActivity {

    private static final String TAG = "DetectionActivityTag";

    private Bitmap mBitmapScan;

    private ImageView ivPreview;
    private ImageView ivPreviewBlur;
    private RecyclerView rvResult;

    private DetectionAdapter mAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detection);

        init();
        setData();
    }

    private void init() {
        //get data from intent
        mBitmapScan = getBitmapScan();

        ivPreview = findViewById(R.id.imageView_detection_preview);
        ivPreviewBlur = findViewById(R.id.imageView_detection_preview_blur);
        rvResult = findViewById(R.id.recycle_detection_result);

        mAdapter = new DetectionAdapter(this, (position, name) -> {
            //clickable
            getDetailBatik(name.toLowerCase().replace("batik ", ""));
        });
    }

    //function for getting batik name
    private void getDetailBatik(String batikName) {
        showLoading(true);

        Call<List<Batik>> call = mApiService.getBatiks(null, batikName, null);
        call.enqueue(new Callback<List<Batik>>() {
            @Override
            public void onResponse(Call<List<Batik>> call, Response<List<Batik>> response) {
                showLoading(false);

                if (response.code() != 200 || response.body() == null) {
                    toast("Failed to load data");
                    return;
                }

                Batik batik = response.body().get(0);

                Intent intent = new Intent(DetectionActivity.this, DetailBatikActivity.class);
                intent.putExtra(Constant.INTENT_BATIK_ID, batik.getId());
                startActivity(intent);
            }

            @Override
            public void onFailure(Call<List<Batik>> call, Throwable t) {
                showLoading(false);
                toast("Failed to load data");
            }
        });
    }

    private Bitmap getBitmapScan() {
        try {
            String imageFromCamera = getIntent().getStringExtra(Constant.INTENT_BITMAP_FROM_CAMERA);
            String imageFromGallery = getIntent().getStringExtra(Constant.INTENT_BITMAP_FROM_GALLERY);

            if (imageFromCamera != null) {
                //image from camera
                return BitmapFactory.decodeStream(openFileInput("RELICBATIK_TEMP_IMAGE"));
            } else {
                //image from gallery
                return MediaStore.Images.Media.getBitmap(this.getContentResolver(), Uri.parse(imageFromGallery));
            }
        } catch (IOException e) {
            return null;
        }
    }

    private void setData() {
        //recycleview
        LinearLayoutManager layoutManager = new LinearLayoutManager(this, RecyclerView.VERTICAL, false);

        rvResult.setLayoutManager(layoutManager);
        rvResult.setNestedScrollingEnabled(false);
        rvResult.setAdapter(mAdapter);

        //validate for preview
        if (mBitmapScan != null) {
            Glide.with(this)
                    .load(mBitmapScan)
                    .into(ivPreview);

            //blur image
            Glide.with(this)
                    .load(mBitmapScan)
                    .apply(RequestOptions.bitmapTransform(new BlurTransformation(90, 3)))
                    .into(ivPreviewBlur);
        }

        //detection
        try {
            Model model = Model.newInstance(this);
            TensorImage image = TensorImage.fromBitmap(mBitmapScan);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(image);
            List<Category> probability = outputs.getProbabilityAsCategoryList();

            probability.sort(Collections.reverseOrder((c1, c2) -> Float.compare(c1.getScore(), c2.getScore())));

            if (probability.size() > 3) {
                mAdapter.addItem(
                        probability.get(0),
                        probability.get(1),
                        probability.get(2)
                );
            } else {
                //update to recycleview
                mAdapter.updateItem(probability);
            }


            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    public void onBackPressed() {
        mBitmapScan = null;
        super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        mBitmapScan = null;
        super.onDestroy();
    }
}