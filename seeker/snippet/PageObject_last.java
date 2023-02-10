//date: 2023-02-10T16:49:34Z
//url: https://api.github.com/gists/d9f89985358b78216fdabb1a4197e858
//owner: https://api.github.com/users/tzkmx

import org.openqa.selenium.By;
import org.openqa.selenium.OutputType;
import org.openqa.selenium.TakesScreenshot;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;

public class PageObject {
    private WebDriver driver;
    private static final Path SCREENSHOTS_DIR = Paths.get("screenshots");

    public PageObject(WebDriver driver) {
        this.driver = driver;
    }

    public void checkPageTitle(String expectedTitle) {
        takeScreenshot("before-checkPageTitle");
        String actualTitle = driver.getTitle();
        assertThat(actualTitle).isEqualTo(expectedTitle);
        takeScreenshot("after-checkPageTitle");
    }

    public void checkElementText(By selector, String expectedText) {
        takeScreenshot("before-checkElementText");
        WebElement element = driver.findElement(selector);
        String actualText = element.getText();
        assertThat(actualText).isEqualTo(expectedText);
        takeScreenshot("after-checkElementText");
    }

    public void checkElementAttribute(By selector, String attribute, String expectedValue) {
        takeScreenshot("before-checkElementAttribute");
        WebElement element = driver.findElement(selector);
        String actualValue = element.getAttribute(attribute);
        assertThat(actualValue).isEqualTo(expectedValue);
        takeScreenshot("after-checkElementAttribute");
    }

    private void takeScreenshot(String name) {
        if (!Files.exists(SCREENSHOTS_DIR)) {
            try {
                Files.createDirectory(SCREENSHOTS_DIR);
            } catch (IOException
