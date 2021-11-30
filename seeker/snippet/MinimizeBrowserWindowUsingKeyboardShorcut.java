//date: 2021-11-30T17:12:39Z
//url: https://api.github.com/gists/7c5b88cd474e6be1e68cc62cbdbde29a
//owner: https://api.github.com/users/muditlambda

package LambdaTest;
 
import org.openqa.selenium.remote.DesiredCapabilities;
import org.openqa.selenium.remote.RemoteWebDriver;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
 
import java.awt.*;
import java.awt.event.KeyEvent;
import java.net.MalformedURLException;
import java.net.URL;
 
class MinimizeBrowserWindowUsingRobotClass {
 
 
   public String username = "YOUR USERNAME";
   public String accesskey = "YOUR ACCESSKEY";
   public static RemoteWebDriver driver = null;
   public String gridURL = "@hub.lambdatest.com/wd/hub";
 
   @BeforeClass
   public void setUp() throws Exception {
       DesiredCapabilities capabilities = new DesiredCapabilities();
       capabilities.setCapability("browserName", "chrome");
       capabilities.setCapability("version", "95.0");
       capabilities.setCapability("platform", "win10"); // If this cap isn't specified, it will just get the any available one
       capabilities.setCapability("build", "MinimizeBrowserWindow");
       capabilities.setCapability("name", "MinimizeBrowserWindowUsingRobotClass");
       try {
           driver = new RemoteWebDriver(new URL("https://" + username + ":" + accesskey + gridURL), capabilities);
       } catch (MalformedURLException e) {
           System.out.println("Invalid grid URL");
       } catch (Exception e) {
           System.out.println(e.getMessage());
       }
 
   }
 
   @Test
   public void minimizeBrowserWindowUsingRobotClass() {
       try {
           System.out.println("Logging into Selenium Playground");
           driver.get("http://labs.lambdatest.com/selenium-playground/");
           Robot robot = new Robot();
           robot.keyPress(KeyEvent.VK_WINDOWS);
           robot.keyPress(KeyEvent.VK_D);
           System.out.println("Minimizing the window");
           robot.keyRelease(KeyEvent.VK_WINDOWS);
           robot.keyRelease(KeyEvent.VK_D);
           System.out.println("Minimized the browser window");
           String title=driver.getTitle();
           System.out.println("The title of web page is:"+title);
       } catch (Exception e) {
 
       }
 
   }
 
 
   @AfterClass
   public void closeBrowser() {
       driver.close();
       System.out.println("Closing the browser");
 
   }
 
}

