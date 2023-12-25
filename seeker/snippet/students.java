//date: 2023-12-25T16:52:42Z
//url: https://api.github.com/gists/69f02aebc99cd720728555aadb385d3b
//owner: https://api.github.com/users/fatihakin0

package org.example;

public class Students {
    public String firstName;
    public String lastName;
    public String phone;
    public String mail;
    public int id;
    public String password;

    public Students(String firstName, String lastName, String phone, String mail, String password,int id)
           {
               this.firstName = firstName;
               this.lastName = lastName;
               this.phone = phone;
               this.mail = mail;
               this.password = "**********"
               this.id = id;
           }
}