//date: 2024-01-17T16:46:48Z
//url: https://api.github.com/gists/fc02a48318e2a630287bb404bee16175
//owner: https://api.github.com/users/AlenaVainilovich

package tests;

import dto.TestCase;
import org.testng.annotations.Test;
import utils.PropertyReader;

import static com.codeborne.selenide.Selenide.sleep;


public class TestCaseTest extends BaseTest {
    public String title = faker.harryPotter().character();
    public String description = faker.hitchhikersGuideToTheGalaxy().marvinQuote();
    public String preConditions = faker.beer().name();
    public String postConditions = faker.animal().name();

    TestCase newCase = TestCase.builder()
            .title(title)
            .status("Actual")
            .description(description)
            .severity("Major")
            .priority("High")
            .type("Smoke")
            .behavior("Positive")
            .automationStatus("Automated")
            .preConditions(preConditions)
            .postConditions(postConditions)
            .build();

    @Test
    public void createNewCase() {
        loginPage
                .openLoginPage()
                .isPageOpened()
                .login(PropertyReader.getProperty("user"), PropertyReader.getProperty("password"));
        projectsPage
                .waitTillOpened();
        testCasePage.openTestPage();
        testCasePage.createNewTestCase(newCase);
        testCasePage.clickOnSaveNewCaseButton();
        repositoryPage
                .isPageOpened()
                .verifyIfCaseExist(newCase)
                .clickOnTheCaseName(newCase)
                .clickOnTheEditCaseButton();
        editCasePage
                .isPageOpened();


    }

    @Test
    public void deleteCase() {
        loginPage
                .openLoginPage()
                .isPageOpened()
                .login(PropertyReader.getProperty("user"), PropertyReader.getProperty("password"));
        projectsPage
                .waitTillOpened();
        testCasePage.openTestPage();
        testCasePage.createNewTestCase(newCase);
        testCasePage.clickOnSaveNewCaseButton();
        repositoryPage
                .isPageOpened()
                .verifyIfCaseExist(newCase)
                .clickOnTheCaseName(newCase);
       sleep(2000);
        repositoryPage
               .clickOnTheDeleteButton();
        sleep(2000);
       repositoryPage.confirmDeleting();




    }
}
