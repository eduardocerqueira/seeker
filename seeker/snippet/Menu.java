//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.model;

import android.graphics.drawable.Drawable;

public class Menu {
    private final String title;
    private final String description;
    private final Drawable icon;

    public Menu(String title, String description, Drawable icon) {
        this.title = title;
        this.description = description;
        this.icon = icon;
    }

    public String getTitle() {
        return title;
    }

    public String getDescription() {
        return description;
    }

    public Drawable getIcon() {
        return icon;
    }
}
