//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.bumptech.glide.load.engine.DiskCacheStrategy;
import com.google.android.material.chip.Chip;
import com.google.android.material.chip.ChipGroup;
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.model.Batik;

import java.util.ArrayList;
import java.util.List;

public class BatikAdapter extends RecyclerView.Adapter<BatikAdapter.BatikHolder> {

    private final Context mContext;
    private final List<Batik> mData;
    private final View mEmptyView;
    private final ClickHandler mHandler;

    public BatikAdapter(Context mContext, View mEmptyView, ClickHandler mHandler) {
        this.mContext = mContext;
        this.mData = new ArrayList<>();
        this.mEmptyView = mEmptyView;
        this.mHandler = mHandler;
    }

    @NonNull
    @Override
    public BatikHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(mContext).inflate(R.layout.list_item_batik, parent, false);
        return new BatikHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull BatikHolder holder, int position) {
        Batik batik = mData.get(position);

        //set data
        holder.tvTitle.setText(batik.getName());

        //set image thumbnail
        Glide.with(mContext)
                .load(batik.getImageUrl())
                .diskCacheStrategy(DiskCacheStrategy.DATA)
                .into(holder.ivThumbnail);

        holder.cgOrigin.removeAllViews();

        //set origin
        for (String origin : batik.getSplitOrigin()) {
            Chip chip = new Chip(mContext);
            chip.setText(origin);
            chip.setChipBackgroundColorResource(R.color.purple_500);
            chip.setTextSize(12);
            chip.setTextAlignment(View.TEXT_ALIGNMENT_CENTER);
            chip.setCheckable(false);
            chip.setCheckedIconVisible(false);
            chip.setTextColor(mContext.getColor(R.color.white));

            holder.cgOrigin.addView(chip);
        }

        //clickable
        holder.itemView.setOnClickListener(view -> {
            mHandler.onItemClicked(batik);
        });
    }

    @Override
    public int getItemCount() {
        return mData.size();
    }

    public void addItems(List<Batik> batiks) {
        mData.clear();
        mData.addAll(batiks);
        notifyDataSetChanged();

        updateView();
    }

    private void updateView() {
        if (mData.size() == 0)
            mEmptyView.setVisibility(View.VISIBLE);
        else
            mEmptyView.setVisibility(View.GONE);
    }

    static class BatikHolder extends RecyclerView.ViewHolder {

        final TextView tvTitle;
        final ImageView ivThumbnail;
        final ChipGroup cgOrigin;

        public BatikHolder(@NonNull View itemView) {
            super(itemView);

            tvTitle = itemView.findViewById(R.id.textView_list_item_batik_title);
            cgOrigin = itemView.findViewById(R.id.chipgroup_list_item_batik_origin);
            ivThumbnail = itemView.findViewById(R.id.imageView_list_item_batik_icon);
        }
    }

    public interface ClickHandler {
        void onItemClicked(Batik batik);
    }
}
