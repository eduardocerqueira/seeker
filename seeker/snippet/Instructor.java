//date: 2023-12-25T16:41:21Z
//url: https://api.github.com/gists/c3190719558d2b073090526830b3ed84
//owner: https://api.github.com/users/elifdev

package org.example;

public class Instructor extends User
{
    private String cv;

    // bir eğitmenin verdiği dersleri nasıl göstermeliyiz? array şeklinde

    public String getCv() {
        return cv;
    }

    public void setCv(String cv) {
        this.cv = cv;
    }

    /*public Instructor(String getUserFirstName, String userLastName, String password, String email, String image, String cv)
    {
        this.cv=cv;
        this.getUserFirstName=getUserFirstName;
    }*/
}