//date: 2026-02-03T17:43:35Z
//url: https://api.github.com/gists/84da2277fb1bd605905eeb5b328d8feb
//owner: https://api.github.com/users/borodicht

package tests;

import org.openqa.selenium.By;
import org.testng.Assert;
import org.testng.annotations.Test;

public class LoginTest extends BaseTest{

    /*
        1. Логин с пустым паролем !!!
        2. Логин с пустым юзером !!!
        3. Логин с невалидными значениями !!!
        4. Позитивный логин !!!
     */

    @Test
    public void checkLogInWithEmptyPassword() {
        loginPage.open();
        loginPage.login("standard_user", "");
        Assert.assertEquals(loginPage.getErrorMessage(), "Epic sadface: "**********"
    }

    @Test
    public void checkLogInWithEmptyUsername() {
        loginPage.open();
        loginPage.login("", "secret_sauce");
        Assert.assertEquals(loginPage.getErrorMessage(), "Epic sadface: Username is required");
    }

    @Test
    public void checkLogInWithNegativeValue() {
        loginPage.open();
        loginPage.login("test","test");
        Assert.assertEquals(loginPage.getErrorMessage(),
                "Epic sadface: "**********"
    }

    @Test
    public void checkLogInWithPositiveValue() {
        loginPage.open();
        loginPage.login("standard_user","secret_sauce");
        Assert.assertTrue(productsPage.titleIsDisplayed());
    }
}
 Assert.assertTrue(productsPage.titleIsDisplayed());
    }
}
