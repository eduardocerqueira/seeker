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
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.model.Menu;

import java.util.List;

public class MenuAdapter extends RecyclerView.Adapter<MenuAdapter.MenuHolder> {

    private final Context mContext;
    private final List<Menu> mData;
    private final ClickHandler mHandler;

    public MenuAdapter(Context mContext, List<Menu> mData, ClickHandler mHandler) {
        this.mContext = mContext;
        this.mData = mData;
        this.mHandler = mHandler;
    }

    @NonNull
    @Override
    public MenuHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(mContext).inflate(R.layout.list_item_menu, parent, false);
        return new MenuHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull MenuHolder holder, int position) {
        Menu menu = mData.get(position);

        holder.tvTitle.setText(menu.getTitle());
        holder.tvDescription.setText(menu.getDescription());
        Glide.with(mContext)
                .load(menu.getIcon())
                .into(holder.ivIcon);

        //clickable
        holder.itemView.setOnClickListener(view -> {
            mHandler.onItemClicked(position, menu);
        });
    }

    @Override
    public int getItemCount() {
        return mData.size();
    }

    static class MenuHolder extends RecyclerView.ViewHolder {
        final TextView tvTitle;
        final TextView tvDescription;
        final ImageView ivIcon;

        public MenuHolder(@NonNull View itemView) {
            super(itemView);
            tvTitle = itemView.findViewById(R.id.textView_list_item_menu_title);
            tvDescription = itemView.findViewById(R.id.textView_list_item_menu_description);
            ivIcon = itemView.findViewById(R.id.imageView_list_item_menu_icon);
        }
    }

    public interface ClickHandler {
        void onItemClicked(int position, Menu menu);
    }
}
