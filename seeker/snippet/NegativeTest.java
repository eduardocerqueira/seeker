//date: 2023-10-25T16:52:16Z
//url: https://api.github.com/gists/e8c0bb0a653c7e1bfbc04600200afb15
//owner: https://api.github.com/users/YanaSinitskaya

@Test
    public void negativeSignUpEmptyFields() {
        System.setProperty("webdriver.chrome.driver", "src/test/resources/chromedriver.exe");
        WebDriver browser = new ChromeDriver();
        browser.get("https://www.sharelane.com/cgi-bin/register.py");
        browser.findElement(By.name("zip_code")).sendKeys("12345");
        browser.findElement(By.cssSelector("[value=Continue]")).click();
        browser.findElement(By.name("first_name")).sendKeys("");
        browser.findElement(By.name("last_name")).sendKeys("");
        browser.findElement(By.name("email")).sendKeys("");
        browser.findElement(By.name("password1")).sendKeys("");
        browser.findElement(By.name("password2")).sendKeys("");
        browser.findElement(By.cssSelector("[value=Register]")).click();
        String errorMessage = browser.findElement(By.cssSelector("[class=error_message]")).getText();
        Assert.assertEquals(errorMessage, "Oops, error on page. Some of your fields have invalid data or email was previously used");
        browser.quit();
    }
}