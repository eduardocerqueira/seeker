//date: 2026-02-03T17:43:35Z
//url: https://api.github.com/gists/84da2277fb1bd605905eeb5b328d8feb
//owner: https://api.github.com/users/borodicht

package pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;

public class LoginPage {

    WebDriver driver;

    By usernameField = By.id("user-name");
    By passwordField = "**********"
    By loginButton = By.id("login-button");
    By errorMessage = By.cssSelector("[data-test='error']");

    public LoginPage(WebDriver driver) {
        this.driver = driver;
    }

    public void open() {
        driver.get("https://www.saucedemo.com/");
    }

    public void login(String user, String password) {
        driver.findElement(usernameField).sendKeys(user);
        driver.findElement(passwordField).sendKeys(password);
        driver.findElement(loginButton).click();
    }

    public String getErrorMessage() {
        return driver.findElement(errorMessage).getText();
    }
}
  }
}
