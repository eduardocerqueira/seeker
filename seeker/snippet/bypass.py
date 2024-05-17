#date: 2024-05-17T17:07:54Z
#url: https://api.github.com/gists/134719f564d7b59f0de303df9d380cb8
#owner: https://api.github.com/users/haris90e

#Approach-1
import org.openqa.selenium.By;
import org.openqa.selenium.Keys;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import java.util.concurrent.TimeUnit;
import org.openqa.selenium.interactions.Action;
import org.openqa.selenium.interactions.Actions;

public class MetdKeyDown{
   public static void main(String[] args) {
System.setProperty("webdriver.chrome.driver","C:\Users\ghs6kor\Desktop\Java\chromedriver.exe");
      WebDriver driver = new ChromeDriver();
      String url = "https://www.example.com/index.htm";
      driver.get(url);
      driver.manage().timeouts().implicitlyWait(4, TimeUnit.SECONDS);
      // identify element
      WebElement l = driver.findElement(By.id("gsc-i-id1"));
      // Actions class
      Actions a = new Actions(driver);
      // moveToElement() and then click()
      a.moveToElement(l).click();
      //enter text with keyDown() SHIFT key ,keyUp() then build() ,perform()
      a.keyDown(Keys.SHIFT);
      a.sendKeys("hello").keyUp(Keys.SHIFT).build().perform();
      driver.quit()
   }
}

#Approach-2
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
import selenium.webdriver.support.expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time

# Install Chrome Driver Manager
service = ChromeService(executable_path=ChromeDriverManager().install())
options = webdriver.ChromeOptions()
# Headless mode
options.add_argument("--headless")
driver = webdriver.Chrome(
    service=service,
    options=options,
)
driver.get("your_url")
# Switch to iframe that directly houses pX "Press & Hold" button
WebDriverWait(driver, timeout=300).until(EC.frame_to_be_available_and_switch_to_it('iframe_name_or_id'))
# Get button element
btn = driver.find_element(By.XPATH, "//xpath_to_button")
# Initialize for low-level interactions
action = ActionChains(driver)
action.click_and_hold(btn)
# Initiate clich and hold action on button
action.perform()
# Keep holding for 10s
time.sleep(10)
# Release button
action.release(btn)

# ...continue scraping

driver.quit()