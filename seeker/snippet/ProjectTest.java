//date: 2024-01-17T16:46:48Z
//url: https://api.github.com/gists/fc02a48318e2a630287bb404bee16175
//owner: https://api.github.com/users/AlenaVainilovich

package tests;

import dto.Project;
import org.testng.annotations.Test;
import utils.PropertyReader;

public class ProjectTest extends BaseTest {
    public String projectName = faker.funnyName().name();
    public String projectCode = faker.currency().code();
    public String description = faker.hitchhikersGuideToTheGalaxy().marvinQuote();

   Project project = Project.builder()
            .projectName(projectName)
            .projectCode(projectCode)
            .description(description)
            .build();
    @Test
    public void projectShouldBeCreated() {
        loginPage.openLoginPage();
        loginPage.login(PropertyReader.getProperty("user"), PropertyReader.getProperty("password"));
        projectsPage.waitTillOpened();
    }

    @Test
    public void createNewProject() {
        loginPage
                .openLoginPage()
                .isPageOpened()
                .login(PropertyReader.getProperty("user"), PropertyReader.getProperty("password"));
        projectsPage
                .waitTillOpened()
                .clickOnCreateNewProjectButton();
        createNewProjectPage
                .isPageOpened()
                .createNewProject(project);
        projectsPage
                .openPage()
                .verifyIsProjectExist(project);






        //projectsPage.newProject();
       // projectsPage.isRepositoryCreated();
    }

}
