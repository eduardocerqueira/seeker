//date: 2022-04-13T17:11:10Z
//url: https://api.github.com/gists/ceec7620684ecb9a49e3a0c422c8a560
//owner: https://api.github.com/users/jeyakeerthanan

package com.company;

public class Application {

    public static void main(String[] args) {
	
        Mobile builder= new Mobile.Builder("A11").airpod("aipod pro").camera("5px").build();
        Mobile.Builder builder2= new Mobile.Builder("snapDragon");
        System.out.println(builder);
        System.out.println(builder2);
    }
}
