//date: 2024-01-17T16:46:48Z
//url: https://api.github.com/gists/fc02a48318e2a630287bb404bee16175
//owner: https://api.github.com/users/AlenaVainilovich

package tests;

import com.codeborne.selenide.Configuration;
import com.codeborne.selenide.logevents.SelenideLogger;
import com.github.javafaker.Faker;
import io.qameta.allure.selenide.AllureSelenide;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import pages.*;

import static com.codeborne.selenide.Selenide.closeWebDriver;


public class BaseTest {

    Faker faker = new Faker();
    LoginPage loginPage;
    ProjectsPage projectsPage;
    CreateNewProjectPage createNewProjectPage;
    TestCasePage testCasePage;
    RepositoryPage repositoryPage;
    EditCasePage editCasePage;


    @BeforeMethod
    public void setup() {
        Configuration.browser = "chrome";
        Configuration.headless = false;
        Configuration.timeout = 10000;
        Configuration.clickViaJs = false;
        Configuration.baseUrl = "https://app.qase.io";
        Configuration.browserSize = "1920x1080";

        SelenideLogger.addListener("AllureSelenide", new AllureSelenide());

        loginPage = new LoginPage();
        projectsPage = new ProjectsPage();
        createNewProjectPage = new CreateNewProjectPage();
        testCasePage = new TestCasePage();
        repositoryPage = new RepositoryPage();
        editCasePage = new EditCasePage();
    }

    @AfterMethod(alwaysRun = true)
    public void close() {
        closeWebDriver();
    }
}
