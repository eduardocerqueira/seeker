//date: 2024-01-17T16:46:48Z
//url: https://api.github.com/gists/fc02a48318e2a630287bb404bee16175
//owner: https://api.github.com/users/AlenaVainilovich

package tests;

import org.testng.annotations.Test;
import utils.PropertyReader;

import static org.testng.Assert.assertEquals;

public class LoginTest extends BaseTest {

    public String invalidUser = faker.internet().emailAddress();
    @Test
    public void login() {
        loginPage
                .openLoginPage()
                .isPageOpened()
                .login(PropertyReader.getProperty("user"), PropertyReader.getProperty("password"));
    }

    @Test
    public void invalidLogin() {
        loginPage
                .openLoginPage()
                .isPageOpened()
                .invalidLogin(invalidUser,PropertyReader.getProperty("password"));
        assertEquals(loginPage.getErrorMessage(), "These credentials do not match our records.");
    }
}
