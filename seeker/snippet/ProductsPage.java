//date: 2026-02-03T17:43:35Z
//url: https://api.github.com/gists/84da2277fb1bd605905eeb5b328d8feb
//owner: https://api.github.com/users/borodicht

package pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;

public class ProductsPage {

    WebDriver driver;
    By title = By.cssSelector("[data-test='title']");

    public ProductsPage(WebDriver driver) {
        this.driver = driver;
    }

    public boolean titleIsDisplayed() {
        return driver.findElement(title).isDisplayed();
    }
}
