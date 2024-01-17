//date: 2024-01-17T16:46:48Z
//url: https://api.github.com/gists/fc02a48318e2a630287bb404bee16175
//owner: https://api.github.com/users/AlenaVainilovich

package wrappers;

import static com.codeborne.selenide.Selenide.$x;

public class ProseMirror {
    String label;
    String locator = "//p[@class='gYZSEd']";

    public ProseMirror(String label) {
        this.label = label;
    }

    public void write(String text) {
        $x((String.format(locator, label))).click();
        $x((String.format(locator, label))).clear();
        $x((String.format(locator, label))).setValue(text);
    }
}
