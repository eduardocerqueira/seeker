//date: 2025-01-09T17:06:25Z
//url: https://api.github.com/gists/780c2128c6c590576e7997dbdd4cfe42
//owner: https://api.github.com/users/borodicht

package ui.drivers;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.firefox.FirefoxOptions;

public class DriverManager {

    private static ThreadLocal <WebDriver> driver = new ThreadLocal<>();

    private DriverManager() {};

    public static WebDriver getDriver() {
        if (driver.get() == null) {
            iniztialiazeDriver();
        }
        return driver.get();
    }

    private static void iniztialiazeDriver() {
        String browser = System.getProperty("browser", "chrome");

        switch (browser.toLowerCase()) {
            case "chrome":
                ChromeOptions options = new ChromeOptions();
                options.addArguments("--headless");
                driver.set(new ChromeDriver(options));
            case "firefox":
                FirefoxOptions options1 = new FirefoxOptions();
                options1.addArguments("--headless");
                driver.set(new FirefoxDriver(options1));
            default:
                throw new IllegalArgumentException(browser);
        }
    }

    public void tearDown() {
        if(driver.get() != null) {
            driver.get().quit();
            driver.remove();
        }
    }
}
