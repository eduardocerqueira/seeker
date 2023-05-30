//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.model;

import com.google.gson.annotations.SerializedName;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class Batik implements Serializable {
    private int id;
    private String name;
    private String origin;
    private String characteristic;
    private String philosophy;

    @SerializedName("batik_class_id")
    private int batikClassId;

    @SerializedName("image_url")
    private String imageUrl;

    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public String getOrigin() {
        return origin;
    }

    public String getCharacteristic() {
        return characteristic;
    }

    public String getPhilosophy() {
        return philosophy;
    }

    public int getBatikClassId() {
        return batikClassId;
    }

    public String getImageUrl() {
        return imageUrl;
    }

    public List<String> getSplitOrigin() {
        List<String> origins = new ArrayList<>();

        for (String s : origin.split(",")) {
            origins.add(s.toUpperCase());
        }

        return origins;
    }
}
