//date: 2024-01-17T16:46:48Z
//url: https://api.github.com/gists/fc02a48318e2a630287bb404bee16175
//owner: https://api.github.com/users/AlenaVainilovich

package pages;

import com.codeborne.selenide.Condition;
import io.qameta.allure.Step;
import lombok.extern.log4j.Log4j2;

import static com.codeborne.selenide.Selenide.$;
import static com.codeborne.selenide.Selenide.open;

@Log4j2
public class LoginPage {

    final String EMAIL_CSS = "[name=email]";
    final String PASSWORD_CSS = "**********"=password]";
    final String SUBMIT_CSS = "[type=submit]";
    final String ERROR_MESSAGE = "//div[@role='alert']//span[@class='ic9QAx']";

    @Step("Login Page opening")
    public LoginPage openLoginPage() {
        log.info("Opening Login Page");
        open("/login");
        return this;
    }

    public LoginPage isPageOpened() {
        $(SUBMIT_CSS).shouldBe(Condition.visible);
        return this;
    }

    @Step("Log in and redirect on Projects Page")
    public ProjectsPage login(String user, String password) {
        $(EMAIL_CSS).sendKeys(user);
        $(PASSWORD_CSS).sendKeys(password);
        $(SUBMIT_CSS).click();
        return new ProjectsPage();
    }

    @Step("Log in with invalid data")
    public LoginPage invalidLogin(String invalidUser, String password) {
        $(EMAIL_CSS).sendKeys(invalidUser);
        $(PASSWORD_CSS).sendKeys(password);
        $(SUBMIT_CSS).click();
        return this;
    }

    @Step("Get error message")
    public String getErrorMessage() {
        String errorMessage = $(ERROR_MESSAGE).getText();
        return errorMessage;
    }


}
