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

public class AccountAdapter extends RecyclerView.Adapter<AccountAdapter.AccountHolder> {
    private final Context mContext;
    private final List<Menu> mData;

    public AccountAdapter(Context mContext, List<Menu> mData) {
        this.mContext = mContext;
        this.mData = mData;
    }

    @NonNull
    @Override
    public AccountHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(mContext).inflate(R.layout.list_item_account, parent, false);
        return new AccountHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull AccountHolder holder, int position) {
        holder.title.setText(mData.get(position).getTitle());
        holder.content.setText(mData.get(position).getDescription());

        Glide.with(mContext)
                .load(mData.get(position).getIcon())
                .into(holder.icon);
    }

    @Override
    public int getItemCount() {
        return mData.size();
    }

    static class AccountHolder extends RecyclerView.ViewHolder {

        final TextView title;
        final TextView content;
        final ImageView icon;

        public AccountHolder(@NonNull View itemView) {
            super(itemView);

            title = itemView.findViewById(R.id.textView_list_item_account_title);
            content = itemView.findViewById(R.id.textView_list_item_account_content);
            icon = itemView.findViewById(R.id.imageView_list_item_account_icon);
        }
    }
}
