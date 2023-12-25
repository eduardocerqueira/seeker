//date: 2023-12-25T16:41:21Z
//url: https://api.github.com/gists/c3190719558d2b073090526830b3ed84
//owner: https://api.github.com/users/elifdev

package org.example;

public class User
{
    private String userFirstName;
    private String userLastName;
    private String password;
    private String email;
    private String image;

    public String getUserFirstName() {
        return userFirstName;
    }

    public void setUserFirstName(String userFirstName) {
        this.userFirstName = userFirstName;
    }

    public String getUserLastName() {
        return userLastName;
    }

    public void setUserLastName(String userLastName) {
        this.userLastName = userLastName;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = "**********"
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getImage() {
        return image;
    }

    public void setImage(String image) {
        this.image = image;
    }
}