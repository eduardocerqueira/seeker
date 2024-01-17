//date: 2024-01-17T16:46:48Z
//url: https://api.github.com/gists/fc02a48318e2a630287bb404bee16175
//owner: https://api.github.com/users/AlenaVainilovich

package wrappers;

import com.codeborne.selenide.Condition;

import static com.codeborne.selenide.Selenide.$x;

public class DropDown {
    String label;
    String labelLocator = "//*[text()='%s']";
    String optionLocator = "//span[@class='YaRdmB']";

    public DropDown(String label) {
        this.label = label;
    }

    public void select(String option) {
        $x(String.format(labelLocator, label)).shouldBe(Condition.visible).click();
        $x(String.format(optionLocator, option)).shouldBe(Condition.visible).click();
    }

}
