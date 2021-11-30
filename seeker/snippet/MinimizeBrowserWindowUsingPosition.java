//date: 2021-11-30T17:06:28Z
//url: https://api.github.com/gists/43659bd4fa84a34b1e902a50be8cefa7
//owner: https://api.github.com/users/muditlambda

package LambdaTest;
 
import org.openqa.selenium.Dimension;
import org.openqa.selenium.Point;
import org.openqa.selenium.remote.DesiredCapabilities;
import org.openqa.selenium.remote.RemoteWebDriver;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
 
import java.net.MalformedURLException;
import java.net.URL;
 
class MinimizeBrowserWindowUsingPosition {
 
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
       capabilities.setCapability("build", "MinimizeBrowserWindowBySettingPosition");
       capabilities.setCapability("name", "MinimizeBrowserWindowBySettingPositionUsingSelenium");
       try {
           driver = new RemoteWebDriver(new URL("https://" + username + ":" + accesskey + gridURL), capabilities);
       } catch (MalformedURLException e) {
           System.out.println("Invalid grid URL");
       } catch (Exception e) {
           System.out.println(e.getMessage());
       }
 
   }
 
   @Test
   public void minimizeBrowserWindowBySettingPosition() {
       try {
           System.out.println("Logging into Selenium Playground");
           driver.get("http://labs.lambdatest.com/selenium-playground/");
           Point p = driver.manage().window().getPosition();
           System.out.println("The Position of the window is:"+p);
           Dimension d = driver.manage().window().getSize();
           System.out.println("The Size of the window is:"+d);
           driver.manage().window().setPosition(new Point((d.getHeight()-p.getX()), (d.getWidth()-p.getY())));
           System.out.println("The New X Coordinate After Minimizing Is:"+(d.getHeight()-p.getX()));
           System.out.println("The New Y Coordinate After Minimizing Is:"+(d.getWidth()-p.getY()));
           System.out.println("Minimized the browser window by setting its position");
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
