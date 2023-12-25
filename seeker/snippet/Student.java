//date: 2023-12-25T16:41:21Z
//url: https://api.github.com/gists/c3190719558d2b073090526830b3ed84
//owner: https://api.github.com/users/elifdev

package org.example;

public class Student extends User
{
    private String address;
    private String creditCard;
    private String phone;

    // bir öğrencinin aldığı dersleri array şeklinde mi göstermeliyiz

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getCreditCard() {
        return creditCard;
    }

    public void setCreditCard(String creditCard) {
        this.creditCard = creditCard;
    }

    public String getPhone() {
        return phone;
    }

    public void setPhone(String phone) {
        this.phone = phone;
    }
}