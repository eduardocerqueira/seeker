//date: 2024-01-17T16:46:48Z
//url: https://api.github.com/gists/fc02a48318e2a630287bb404bee16175
//owner: https://api.github.com/users/AlenaVainilovich

package pages;

import com.codeborne.selenide.Condition;
import dto.Project;
import lombok.extern.log4j.Log4j2;

import static com.codeborne.selenide.Selenide.$;

@Log4j2
public class CreateNewProjectPage extends BasePage{
    final String PROJECT_NAME_CSS = "#project-name";
    final String PROJECT_CODE_CSS = "#project-code";
    final String DESCRIPTION_CSS = "#description-area";
    final String CREATE_PROJECT_BUTTON_CSS = "[type=submit]";

    public CreateNewProjectPage isPageOpened() {
        $(CREATE_PROJECT_BUTTON_CSS).shouldBe(Condition.visible);
        return this;
    }

    public ProjectsPage createNewProject(Project project) {
        $(PROJECT_NAME_CSS).clear();
        $(PROJECT_CODE_CSS).setValue(project.getProjectName());
        $(DESCRIPTION_CSS).clear();
        $(PROJECT_NAME_CSS).setValue(project.getProjectCode());
        $(PROJECT_CODE_CSS).clear();
        $(DESCRIPTION_CSS).setValue(project.getDescription());
        $(CREATE_PROJECT_BUTTON_CSS).click();
        return new ProjectsPage();
    }
}
