//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.model;

import com.google.gson.annotations.SerializedName;

import java.io.Serializable;

public class Store implements Serializable {
    private int id;
    private String name;
    private String description;
    private String address;
    private String latitude;
    private String longitude;
    private double distance;

    @SerializedName("operational_hour")
    private String operationalHour;

    @SerializedName("thumbnail_url")
    private String thumbnailUrl;

    private String telephone;

    @SerializedName("user_id")
    private int userId;

    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public String getAddress() {
        return address;
    }

    public String getLatitude() {
        return latitude;
    }

    public String getLongitude() {
        return longitude;
    }

    public String getOperationalHour() {
        return operationalHour;
    }

    public String getThumbnailUrl() {
        return thumbnailUrl;
    }

    public String getTelephone() {
        return telephone;
    }

    public int getUserId() {
        return userId;
    }

    public double getDistance() {
        return distance;
    }
}
