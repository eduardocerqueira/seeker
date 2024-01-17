//date: 2024-01-17T16:46:48Z
//url: https://api.github.com/gists/fc02a48318e2a630287bb404bee16175
//owner: https://api.github.com/users/AlenaVainilovich

package pages;

import com.codeborne.selenide.Condition;
import dto.Project;
import io.qameta.allure.Step;
import lombok.extern.log4j.Log4j2;
import org.openqa.selenium.By;

import static com.codeborne.selenide.Selenide.*;
import static org.testng.Assert.assertEquals;

@Log4j2
public class ProjectsPage {

    public final String PROJECT_NAME = "//a[contains(text(),'%s')]";
    public final String CREATE_NEW_PROJECT_BUTTON = "#createButton";

    @Step("Open 'Project Page'")
    public ProjectsPage openPage() {
        open("/projects");
        return this;
    }

    public ProjectsPage waitTillOpened() {
        $(CREATE_NEW_PROJECT_BUTTON).shouldBe(Condition.visible);
        return this;
    }

    public CreateNewProjectPage clickOnCreateNewProjectButton() {
        $(CREATE_NEW_PROJECT_BUTTON).click();
    return new CreateNewProjectPage();
    }

    public ProjectsPage verifyIsProjectExist(Project project) {
        $x(String.format(PROJECT_NAME, project.getProjectCode())).shouldBe(Condition.visible);
        return this;
    }




    public void isRepositoryCreated() {
        assertEquals("BLA repository", $(By.xpath("//*[text()=' repository']")).getText());
    }

}
