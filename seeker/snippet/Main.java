//date: 2023-06-05T16:56:23Z
//url: https://api.github.com/gists/01452f2de81ae849c319217c681d4009
//owner: https://api.github.com/users/CherevkoSV

package org.example;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        System.setProperty("webdriver.chrome.driver", "/Users/anymacstore/Downloads/chromedriver_mac64/chromedriver");
        WebDriver driver = new ChromeDriver();

        driver.get("https://ithillel.ua/");

        WebElement courseButton = driver.findElement(By.cssSelector("#body > div.site-wrapper > main > section.section.-courses > div > div > div.courses-section_cats > div > ul > li:nth-child(1) > a > div"));
        courseButton.click();

        WebElement basicFrontEndCourseButton = driver.findElement(By.cssSelector("#categories > div.profession > div > ul > li.profession_item.-active > div > div:nth-child(1) > ul > li:nth-child(1) > a > div.profession-bar_body"));
        basicFrontEndCourseButton.click();

        List<WebElement> teachersNames = driver.findElements(By.cssSelector("#coachesSection > div > div > ul"));
        for (WebElement teacherName : teachersNames) {
            System.out.println(teacherName.getText());
        }

        driver.quit();
    }
}
