//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.ui.activity;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;

import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.load.engine.DiskCacheStrategy;
import com.otaliastudios.zoom.ZoomImageView;
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.base.BaseActivity;
import com.suncode.relicbatik.base.Constant;

import java.util.Objects;

public class ImagePreviewActivity extends BaseActivity {

    private String mImageUrl;

    private ZoomImageView mImageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_preview);

        init();
        setData();
    }

    private void init() {
        //set navigation bar and status color to black
        getWindow().setStatusBarColor(Color.BLACK);
        Objects.requireNonNull(getSupportActionBar()).setBackgroundDrawable(new ColorDrawable(Color.BLACK));

        mImageUrl = getIntent().getStringExtra(Constant.INTENT_IMAGE_URL);
        mImageView = findViewById(R.id.imageView_preview_image);
    }

    private void setData() {
        Glide.with(this)
                .load(mImageUrl)
                .error(ContextCompat.getDrawable(this, R.drawable.ic_batik))
                .diskCacheStrategy(DiskCacheStrategy.DATA)
                .into(mImageView);
    }
}