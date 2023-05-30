//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.model;

import com.google.gson.annotations.SerializedName;

import java.io.Serializable;

public class StoreImage implements Serializable {
    private int id;

    @SerializedName("image_url")
    private String imageUrl;

    @SerializedName("seller_store_id")
    private int sellerStoreId;

    public int getId() {
        return id;
    }

    public String getImageUrl() {
        return imageUrl;
    }

    public int getSellerStoreId() {
        return sellerStoreId;
    }
}
