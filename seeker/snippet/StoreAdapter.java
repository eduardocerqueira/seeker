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
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.bumptech.glide.load.engine.DiskCacheStrategy;
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.model.Store;

import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.List;

public class StoreAdapter extends RecyclerView.Adapter<StoreAdapter.StoreHolder> {

    private final Context mContext;
    private final List<Store> mData;
    private final View mEmptyView;
    private final ClickHandler mHandler;

    public StoreAdapter(Context mContext, View mEmptyView, ClickHandler mHandler) {
        this.mContext = mContext;
        this.mData = new ArrayList<>();
        this.mEmptyView = mEmptyView;
        this.mHandler = mHandler;
    }

    @NonNull
    @Override
    public StoreHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(mContext).inflate(R.layout.list_item_store, parent, false);
        return new StoreHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull StoreHolder holder, int position) {
        Store store = mData.get(position);

        //setup component
        holder.tvTitle.setText(store.getName());
        holder.tvDistance.setText(MessageFormat.format("{0}Km", store.getDistance()));

        Glide.with(mContext)
                .load(store.getThumbnailUrl())
                .diskCacheStrategy(DiskCacheStrategy.DATA)
                .error(ContextCompat.getDrawable(mContext, R.drawable.ic_batik))
                .into(holder.ivThumbnail);

        //clickable
        holder.itemView.setOnClickListener(view -> {
            mHandler.onItemClicked(store);
        });
    }

    @Override
    public int getItemCount() {
        return mData.size();
    }

    public void addItems(List<Store> stores) {
        this.mData.clear();
        this.mData.addAll(stores);
        notifyDataSetChanged();

        updateView();
    }

    private void updateView() {
        if (mData.size() == 0)
            mEmptyView.setVisibility(View.VISIBLE);
        else
            mEmptyView.setVisibility(View.GONE);
    }

    static class StoreHolder extends RecyclerView.ViewHolder {

        final TextView tvTitle;
        final TextView tvDistance;
        final ImageView ivThumbnail;

        public StoreHolder(@NonNull View itemView) {
            super(itemView);

            tvTitle = itemView.findViewById(R.id.textView_list_item_store_title);
            tvDistance = itemView.findViewById(R.id.textView_list_item_store_content);
            ivThumbnail = itemView.findViewById(R.id.imageView_list_item_store_icon);
        }
    }

    public interface ClickHandler {
        void onItemClicked(Store store);
    }
}
