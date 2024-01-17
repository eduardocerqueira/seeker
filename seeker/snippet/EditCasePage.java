//date: 2024-01-17T16:46:48Z
//url: https://api.github.com/gists/fc02a48318e2a630287bb404bee16175
//owner: https://api.github.com/users/AlenaVainilovich

package pages;

import com.codeborne.selenide.Condition;

import static com.codeborne.selenide.Selenide.$x;

public class EditCasePage {
    public final String EDIT_CASE_PAGE_HEADER = "//h1[text()='Edit test case']";
    public EditCasePage isPageOpened() {
        $x(EDIT_CASE_PAGE_HEADER).shouldBe(Condition.visible);
        return this;
    }


}
