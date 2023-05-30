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

import com.suncode.relicbatik.R;

import org.tensorflow.lite.support.label.Category;

import java.text.DecimalFormat;
import java.text.MessageFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DetectionAdapter extends RecyclerView.Adapter<DetectionAdapter.DetectionHolder> {

    private final Context mContext;
    private final List<Category> mData;
    private final ClickHandler mHandler;

    public DetectionAdapter(Context mContext, ClickHandler mHandler) {
        this.mContext = mContext;
        this.mHandler = mHandler;
        this.mData = new ArrayList<>();
    }

    @NonNull
    @Override
    public DetectionHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(mContext).inflate(R.layout.list_item_detection, parent, false);
        return new DetectionHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull DetectionHolder holder, int position) {
        Category category = mData.get(position);

        holder.tvName.setText(category.getLabel());
        holder.tvScore.setText(MessageFormat.format("{0}%", Math.round(category.getScore() * 100)));

        //validate background colour and clickable
        if (position != 0) {
            holder.vBox.setBackgroundColor(mContext.getColor(android.R.color.darker_gray));
        }

        if (position == 0) {
            holder.itemView.setOnClickListener(view -> {
                mHandler.onItemClicked(position, category.getLabel());
            });
            holder.vDetail.setVisibility(View.VISIBLE);
        } else {
            holder.vDetail.setVisibility(View.GONE);
        }
    }

    @Override
    public int getItemCount() {
        return mData.size();
    }

    public void addItem(Category... categories) {
        this.mData.clear();
        this.mData.addAll(Arrays.asList(categories));

        notifyDataSetChanged();
    }

    public void updateItem(List<Category> categories) {
        this.mData.clear();
        this.mData.addAll(categories);

        notifyDataSetChanged();
    }

    static class DetectionHolder extends RecyclerView.ViewHolder {

        final TextView tvName;
        final TextView tvScore;
        final View vDetail;
        final View vBox;

        public DetectionHolder(@NonNull View itemView) {
            super(itemView);

            tvName = itemView.findViewById(R.id.textview_list_item_detection_name);
            tvScore = itemView.findViewById(R.id.textview_list_item_detection_score);
            vDetail = itemView.findViewById(R.id.view_list_item_detection_detail);
            vBox = itemView.findViewById(R.id.view_list_item_detection);
        }
    }

    public interface ClickHandler {
        void onItemClicked(int position, String name);
    }
}
