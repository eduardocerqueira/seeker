//date: 2023-10-25T16:49:55Z
//url: https://api.github.com/gists/5726e83a4151a3ab066f7c933b3ffe43
//owner: https://api.github.com/users/AlenaVainilovich

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.testng.Assert;
import org.testng.annotations.Test;

public class SignUpTest {

    @Test
    public void zipCode4Digits() {
        //Открыть браузер
        //Открыть страницу https://www.sharelane.com/cgi-bin/register.py
        // ввести 4 цифры, например 1111
        // Нажать на кнопку Continue
        //Проверить наличие ошибки "Oops, error on page. ZIP code should have 5 digits
        //Закрыть браузер
        System.setProperty("webdriver.chrome.driver", "src/test/resources/chromedriver.exe");
        WebDriver driver = new ChromeDriver();
        driver.get("https://www.sharelane.com/cgi-bin/register.py");
        driver.findElement(By.name("zip_code")).sendKeys("1111");
        driver.findElement(By.cssSelector("[value = Continue]")).click();
        String error = driver.findElement(By.cssSelector("[class=error_message]")).getText();
        Assert.assertEquals(error, "Oops, error on page. ZIP code should have 5 digits");
        driver.quit();
    }

    @Test
    public void redirectOnNextStepValidZipCodeInput() {
        System.setProperty("webdriver.chrome.driver", "src/test/resources/chromedriver.exe");
        WebDriver driver = new ChromeDriver();
        driver.get("https://www.sharelane.com/cgi-bin/register.py");
        driver.findElement(By.name("zip_code")).sendKeys("11111");
        driver.findElement(By.cssSelector("[value = Continue]")).click();
        boolean isOpenedNextStep = driver.findElement(By.cssSelector("[value = Register]")).isDisplayed();
        Assert.assertTrue(isOpenedNextStep, "Oops, error on page. ZIP code should have 5 digits");
        driver.quit();
    }

    @Test
    public void errorMessageOnExcessiveZipCodeDigits() {
        System.setProperty("webdriver.chrome.driver", "src/test/resources/chromedriver.exe");
        WebDriver driver = new ChromeDriver();
        driver.get("https://www.sharelane.com/cgi-bin/register.py");
        driver.findElement(By.name("zip_code")).sendKeys("111111");
        driver.findElement(By.cssSelector("[value = Continue]")).click();
        String error = driver.findElement(By.cssSelector("[class=error_message]")).getText();
        Assert.assertEquals(error, "Oops, error on page. ZIP code should have 5 digits");
        driver.quit();
    }

    @Test
    public void errorMessageOnLettersInZipCode() {
        System.setProperty("webdriver.chrome.driver", "src/test/resources/chromedriver.exe");
        WebDriver driver = new ChromeDriver();
        driver.get("https://www.sharelane.com/cgi-bin/register.py");
        driver.findElement(By.name("zip_code")).sendKeys("QwErT");
        driver.findElement(By.cssSelector("[value = Continue]")).click();
        String error = driver.findElement(By.cssSelector("[class=error_message]")).getText();
        Assert.assertEquals(error, "Oops, error on page. ZIP code should have 5 digits");
        driver.quit();
    }

    @Test
    public void errorMessageOnSymbolsInZipCode() {
        System.setProperty("webdriver.chrome.driver", "src/test/resources/chromedriver.exe");
        WebDriver driver = new ChromeDriver();
        driver.get("https://www.sharelane.com/cgi-bin/register.py");
        driver.findElement(By.name("zip_code")).sendKeys("!@#$%");
        driver.findElement(By.cssSelector("[value = Continue]")).click();
        String error = driver.findElement(By.cssSelector("[class=error_message]")).getText();
        Assert.assertEquals(error, "Oops, error on page. ZIP code should have 5 digits");
        driver.quit();
    }

    @Test
    public void zipCodeFieldIsRequired() {
        System.setProperty("webdriver.chrome.driver", "src/test/resources/chromedriver.exe");
        WebDriver driver = new ChromeDriver();
        driver.get("https://www.sharelane.com/cgi-bin/register.py");
        driver.findElement(By.name("zip_code")).sendKeys("");
        driver.findElement(By.cssSelector("[value = Continue]")).click();
        String error = driver.findElement(By.cssSelector("[class=error_message]")).getText();
        Assert.assertEquals(error, "Oops, error on page. ZIP code should have 5 digits");
        driver.quit();
    }

    @Test
    public void positiveSignUp() {
        System.setProperty("webdriver.chrome.driver", "src/test/resources/chromedriver.exe");
        WebDriver driver = new ChromeDriver();
        driver.get("https://www.sharelane.com/cgi-bin/register.py");
        driver.findElement(By.name("zip_code")).sendKeys("11111");
        driver.findElement(By.cssSelector("[value = Continue]")).click();
        driver.findElement(By.name("first_name")).sendKeys("Joe");
        driver.findElement(By.name("last_name")).sendKeys("Doe");
        driver.findElement(By.name("email")).sendKeys("qwertys@tut.by");
        driver.findElement(By.name("password1")).sendKeys("123456");
        driver.findElement(By.name("password2")).sendKeys("123456");
        driver.findElement(By.cssSelector("[value = Register]")).click();
        String signUpMessage = driver.findElement(By.cssSelector("[class=confirmation_message]")).getText();
        Assert.assertEquals(signUpMessage, "Account is created!");
        driver.quit();
    }

    @Test
    public void signUpWithEmptyFields() {
        System.setProperty("webdriver.chrome.driver", "src/test/resources/chromedriver.exe");
        WebDriver driver = new ChromeDriver();
        driver.get("https://www.sharelane.com/cgi-bin/register.py");
        driver.findElement(By.name("zip_code")).sendKeys("11111");
        driver.findElement(By.cssSelector("[value = Continue]")).click();
        driver.findElement(By.cssSelector("[value = Register]")).click();
        String signUpMessage = driver.findElement(By.cssSelector("[class=error_message]")).getText();
        Assert.assertEquals(signUpMessage, "Oops, error on page. Some of your fields have invalid data or email was previously used");
        driver.quit();
    }
}
