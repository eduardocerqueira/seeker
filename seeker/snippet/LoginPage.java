//date: 2023-11-13T17:09:39Z
//url: https://api.github.com/gists/dcad67b2ccdfbfd22faa4d4f8815b857
//owner: https://api.github.com/users/AlenaVainilovich

package pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;

public class LoginPage {

    WebDriver driver;
    private final By UERNAME_INPUT = By.id("user-name");
    private final By PASSWORD_INPUT = "**********"
    private final By LOGIN_INPUT = "**********"
    private final By ERROR_MESSAGE = By.cssSelector("[data-test='error']");

    public LoginPage(WebDriver driver) {
        this.driver = driver;
    }

    public void openPage() {
        driver.get("https://www.saucedemo.com");
    }

    public void login(String user, String password) {
        driver.findElement(UERNAME_INPUT).sendKeys(user);
        driver.findElement(PASSWORD_INPUT).sendKeys(password);
        driver.findElement(LOGIN_INPUT).click();
    }

    public String getErrorMessage() {
        return driver.findElement(ERROR_MESSAGE).getText();
    }
}
();
    }
}
