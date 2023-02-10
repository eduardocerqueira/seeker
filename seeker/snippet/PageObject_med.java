//date: 2023-02-10T16:49:34Z
//url: https://api.github.com/gists/d9f89985358b78216fdabb1a4197e858
//owner: https://api.github.com/users/tzkmx

public class PageObject {
    private WebDriver driver;

    public PageObject(WebDriver driver) {
        this.driver = driver;
    }

    public void checkPageTitle(String expectedTitle) {
        String actualTitle = driver.getTitle();
        assertThat(actualTitle).isEqualTo(expectedTitle);
    }

    public void checkElementText(By selector, String expectedText) {
        WebElement element = driver.findElement(selector);
        String actualText = element.getText();
        assertThat(actualText).isEqualTo(expectedText);
    }

    public void checkElementAttribute(By selector, String attribute, String expectedValue) {
        WebElement element = driver.findElement(selector);
        String actualValue = element.getAttribute(attribute);
        assertThat(actualValue).isEqualTo(expectedValue);
    }
}
