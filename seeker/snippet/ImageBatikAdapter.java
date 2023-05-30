//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.bumptech.glide.load.engine.DiskCacheStrategy;
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.model.ImageBatik;

import java.util.ArrayList;
import java.util.List;

public class ImageBatikAdapter extends RecyclerView.Adapter<ImageBatikAdapter.ImageBatikHolder> {

    private final Context mContext;
    private final List<ImageBatik> mData;
    private final View mEmptyview;
    private final ClickHandler mHandler;

    public ImageBatikAdapter(Context mContext, View mEmptyview, ClickHandler mHandler) {
        this.mContext = mContext;
        this.mData = new ArrayList<>();
        this.mEmptyview = mEmptyview;
        this.mHandler = mHandler;
    }

    @NonNull
    @Override
    public ImageBatikHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(mContext).inflate(R.layout.list_item_image_batik, parent, false);
        return new ImageBatikHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ImageBatikHolder holder, int position) {
        Glide.with(mContext)
                .load(mData.get(position).getImageUrl())
                .diskCacheStrategy(DiskCacheStrategy.DATA)
                .into(holder.ivPreview);

        holder.itemView.setOnClickListener(view -> {
            mHandler.onItemClicked(mData.get(position).getImageUrl());
        });
    }

    @Override
    public int getItemCount() {
        return mData.size();
    }

    public void addItems(List<ImageBatik> imageBatiks) {
        this.mData.clear();
        this.mData.addAll(imageBatiks);
        notifyDataSetChanged();

        updateEmptyView();
    }

    private void updateEmptyView() {
        if (mData.size() == 0)
            mEmptyview.setVisibility(View.VISIBLE);
        else
            mEmptyview.setVisibility(View.GONE);
    }

    static class ImageBatikHolder extends RecyclerView.ViewHolder {
        final ImageView ivPreview;

        public ImageBatikHolder(@NonNull View itemView) {
            super(itemView);

            ivPreview = itemView.findViewById(R.id.imageView_list_item_image_batik_preview);
        }
    }

    public interface ClickHandler {
        void onItemClicked(String url);
    }
}
