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
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.model.TopBatikLike;

import java.util.ArrayList;
import java.util.List;

public class TopBatikLikeAdapter extends RecyclerView.Adapter<TopBatikLikeAdapter.TopBatikLikeHolder> {

    private final Context mContext;
    private final List<TopBatikLike> mData;
    private final View mEmptyView;
    private final ClickHandler mHandler;

    public TopBatikLikeAdapter(Context mContext, View mEmptyView, ClickHandler mHandler) {
        this.mContext = mContext;
        this.mData = new ArrayList<>();
        this.mEmptyView = mEmptyView;
        this.mHandler = mHandler;
    }

    @NonNull
    @Override
    public TopBatikLikeHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(mContext).inflate(R.layout.list_item_top_batik_like, parent, false);
        return new TopBatikLikeHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull TopBatikLikeHolder holder, int position) {
        TopBatikLike data = mData.get(position);

        holder.title.setText(data.getName());
        holder.content.setText(data.getTotal() + " Like");

        Glide.with(mContext)
                .load(data.getImageUrl())
                .diskCacheStrategy(DiskCacheStrategy.DATA)
                .into(holder.icon);

        //clickhandler
        holder.itemView.setOnClickListener(v -> mHandler.onItemClicked(position, data));
    }

    @Override
    public int getItemCount() {
        return mData.size();
    }

    public void addItem(List<TopBatikLike> topBatikLikes) {
        this.mData.clear();
        this.mData.addAll(topBatikLikes);
        notifyDataSetChanged();

        updateView();
    }

    private void updateView() {
        if (mData.size() > 0)
            mEmptyView.setVisibility(View.GONE);
        else
            mEmptyView.setVisibility(View.VISIBLE);
    }

    static class TopBatikLikeHolder extends RecyclerView.ViewHolder {

        final TextView title;
        final TextView content;
        final ImageView icon;

        public TopBatikLikeHolder(@NonNull View itemView) {
            super(itemView);

            title = itemView.findViewById(R.id.textView_list_item_top_batik_like_title);
            content = itemView.findViewById(R.id.textView_list_item_top_batik_like_content);
            icon = itemView.findViewById(R.id.imageView_list_item_top_batik_like_icon);
        }
    }

    public interface ClickHandler {
        void onItemClicked(int position, TopBatikLike topBatikLike);
    }
}
