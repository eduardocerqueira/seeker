//date: 2023-12-25T16:52:42Z
//url: https://api.github.com/gists/69f02aebc99cd720728555aadb385d3b
//owner: https://api.github.com/users/fatihakin0

package org.example;

public class Titles {
   public String titleName;
   public String titleInstructor;
   public String titleDescription;
   public int titlePrice;
   public String  titleimageURL;


    public Titles(String titleName, String titleInstructor, String titleDescription,int titlePrice,String titleimageURL)
    {
        this.titleName = titleName;
        this.titleInstructor = titleInstructor;
        this.titleDescription = titleDescription;
        this.titlePrice = titlePrice;
        this.titleimageURL = titleimageURL;
    }

}