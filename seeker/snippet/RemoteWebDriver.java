//date: 2022-01-25T16:53:36Z
//url: https://api.github.com/gists/aa3e09ec6c098d5362eada598e824068
//owner: https://api.github.com/users/muditlambda

package LambdaTest;

import java.net.MalformedURLException;
import java.net.URL;

import org.openqa.selenium.By;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.remote.CapabilityType;
import org.openqa.selenium.remote.DesiredCapabilities;
import org.openqa.selenium.remote.RemoteWebDriver;
import org.testng.annotations.AfterTest;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class TestCaseRunToGrid {
	

	  public  RemoteWebDriver driver = null;

	 
	  @BeforeTest
	  public void setUp() throws Exception {

	      ChromeOptions options = new ChromeOptions();
	       options.setAcceptInsecureCerts(true);
	       options.setCapability(CapabilityType.BROWSER_NAME,"chrome");
	    
	      try {
	    	   driver = new RemoteWebDriver(new URL("http://localhost:4444"), options);
	         
	      } catch (MalformedURLException e) {
	          System.out.println("Invalid grid URL");
	      } catch (Exception e) {
	          System.out.println(e.getMessage());
	      }
	  }
	 
	  @Test
	  public void firstTestCase() {
	      try {
	          System.out.println("Logging into Lambda Test Selenium PlayGround page ");
	          driver.get("https://www.lambdatest.com/selenium-playground/simple-form-demo");
	          WebElement messageText=driver.findElement(By.cssSelector("input#user-message"));
	          messageTextBox.sendKeys("Welcome to cloud grid");
	          
	          WebElement getValueButton = driver.findElement(By.cssSelector("#showInput"));
	          getValueButton.click();
	          System.out.println("Clicked on the Get Checked Value button");
	      } catch (Exception e) {
	 
	      }
	   }
	 
	  @AfterTest
	  public void closeBrowser() {
	      driver.close();
	      System.out.println("The driver has been closed.");
	  }

}
