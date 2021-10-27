//date: 2021-10-27T16:55:09Z
//url: https://api.github.com/gists/fdc8f4190a8d0cb01c3fe132ece9c5d3
//owner: https://api.github.com/users/wojciechlibor

package com.example.twjkoszyk;

import android.content.Context;
import android.graphics.Paint;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.CompoundButton;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;

public class ShoppingAdapter extends RecyclerView.Adapter<ShoppingAdapter.ProductViewHolder> {

    private ArrayList<Product> products;
    private LayoutInflater inflater;

    public ShoppingAdapter(Context context, ArrayList<Product> products) {
        this.products = products;
        inflater = LayoutInflater.from(context);
    }

    public void addProduct(Product prod){
        products.add(prod);
        notifyItemInserted(products.size()-1);  //informujemy adapter ze na koncu jest nowy element

    }

    public void delProduct(){
        boolean b = products.removeIf(x-> x.isChecked());
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ProductViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View proditemView = inflater.inflate(R.layout.item_product, parent, false);

        return new ProductViewHolder(proditemView, this);
    }

    @Override
    public void onBindViewHolder(@NonNull ProductViewHolder holder, int position) {
        Product current_prod = products.get(position);
        holder.checkBoxItem.setText(current_prod.getName());
        holder.checkBoxItem.setChecked(current_prod.isChecked());

        holder.checkBoxItem.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                current_prod.setChecked(b);
                if (b){
                    holder.checkBoxItem.setPaintFlags(holder.checkBoxItem.getPaintFlags() | Paint.STRIKE_THRU_TEXT_FLAG);
                }
                else{
                    holder.checkBoxItem.setPaintFlags(holder.checkBoxItem.getPaintFlags() & ~Paint.STRIKE_THRU_TEXT_FLAG);
                }
            }
        });

    }

    @Override
    public int getItemCount() {
        return products.size();
    }




    //podklasa
    public class ProductViewHolder extends RecyclerView.ViewHolder {
        public CheckBox checkBoxItem;
        final ShoppingAdapter shopAdapter;


        public ProductViewHolder(@NonNull View itemView, ShoppingAdapter shopAdapter) {
            super(itemView);
            this.shopAdapter = shopAdapter;
            checkBoxItem = itemView.findViewById(R.id.checkBox);
        }
    }//koniec podklasy
}
