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
import com.suncode.relicbatik.model.StoreImage;

import java.util.ArrayList;
import java.util.List;

public class StoreImageAdapter extends RecyclerView.Adapter<StoreImageAdapter.StoreImageHolder> {

    private final Context mContext;
    private final List<StoreImage> mData;
    private final ClickHanlder mHandler;

    public StoreImageAdapter(Context mContext, ClickHanlder mHandler) {
        this.mContext = mContext;
        this.mData = new ArrayList<>();
        this.mHandler = mHandler;
    }

    @NonNull
    @Override
    public StoreImageHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(mContext).inflate(R.layout.list_item_store_image, parent, false);
        return new StoreImageHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull StoreImageHolder holder, int position) {
        StoreImage image = mData.get(position);

        Glide.with(mContext)
                .load(image.getImageUrl())
                .diskCacheStrategy(DiskCacheStrategy.DATA)
                .into(holder.ivImage);

        //clickable
        holder.itemView.setOnClickListener(view -> {
            mHandler.onItemClicked(image.getImageUrl());
        });
    }

    @Override
    public int getItemCount() {
        return mData.size();
    }

    public void addItems(List<StoreImage> storeImages) {
        this.mData.clear();
        this.mData.addAll(storeImages);
        notifyDataSetChanged();
    }

    static class StoreImageHolder extends RecyclerView.ViewHolder {

        final ImageView ivImage;

        public StoreImageHolder(@NonNull View itemView) {
            super(itemView);
            ivImage = itemView.findViewById(R.id.imageview_list_item_store_image);
        }
    }

    public interface ClickHanlder {
        void onItemClicked(String url);
    }
}
