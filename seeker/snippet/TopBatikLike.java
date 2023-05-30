//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.model;

import com.google.gson.annotations.SerializedName;

import java.io.Serializable;

public class TopBatikLike implements Serializable {
    @SerializedName("batik_id")
    private int batikId;

    private String name;

    @SerializedName("image_batik_id")
    private String imageBatikId;

    @SerializedName("image_url")
    private String imageUrl;

    private int total;

    public int getBatikId() {
        return batikId;
    }

    public String getName() {
        return name;
    }

    public String getImageBatikId() {
        return imageBatikId;
    }

    public String getImageUrl() {
        return imageUrl;
    }

    public int getTotal() {
        return total;
    }
}
