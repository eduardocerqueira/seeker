//date: 2021-10-27T16:55:09Z
//url: https://api.github.com/gists/fdc8f4190a8d0cb01c3fe132ece9c5d3
//owner: https://api.github.com/users/wojciechlibor

package com.example.twjkoszyk;

public class Product {

    private String name;
    private boolean isChecked;

    public Product(String name) {
        this.name = name;
        this.isChecked = false;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public boolean isChecked() {
        return isChecked;
    }

    public void setChecked(boolean checked) {
        isChecked = checked;
    }
}
