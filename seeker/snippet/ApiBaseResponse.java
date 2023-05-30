//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.model;

import java.io.Serializable;

public class ApiBaseResponse implements Serializable {
    private boolean status;
    private String messsage;

    public boolean isStatus() {
        return status;
    }

    public String getMesssage() {
        return messsage;
    }
}
