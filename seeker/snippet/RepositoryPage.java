//date: 2024-01-17T16:46:48Z
//url: https://api.github.com/gists/fc02a48318e2a630287bb404bee16175
//owner: https://api.github.com/users/AlenaVainilovich

package pages;

import com.codeborne.selenide.Condition;
import dto.TestCase;

import static com.codeborne.selenide.Selenide.$;


public class RepositoryPage {
    public final String CREATE_CASE_BUTTON = "#create-case-button";
    public final String CASE_NAME = ".wq7uNh";
    public final String EDIT_CASE_BUTTON =".fa-pencil";
    public final String DELETE_CASE_BUTTON =".fa-trash";
    public final String CONFIRM_DELETE = ".b_jd28";

    public RepositoryPage isPageOpened() {
        $(CREATE_CASE_BUTTON).shouldBe(Condition.visible);
        return this;
    }

    public RepositoryPage verifyIfCaseExist(TestCase testCase) {
        $(String.format(CASE_NAME, testCase.getTitle())).shouldBe(Condition.visible);
        return this;
    }

    public RepositoryPage clickOnTheCaseName(TestCase testCase) {
        $(String.format(CASE_NAME, testCase.getTitle())).click();
        return this;
    }

    public EditCasePage clickOnTheEditCaseButton () {
        $(EDIT_CASE_BUTTON).click();
        return new EditCasePage();
    }

    public void clickOnTheDeleteButton() {
        $(DELETE_CASE_BUTTON).click();
    }

    public void confirmDeleting() {
        $(CONFIRM_DELETE).click();
    }
}
