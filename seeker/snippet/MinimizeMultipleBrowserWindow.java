//date: 2021-11-30T17:15:01Z
//url: https://api.github.com/gists/1fbcd1f992fc8fd81440476180314b0d
//owner: https://api.github.com/users/muditlambda

package LambdaTest;
 
import org.openqa.selenium.remote.DesiredCapabilities;
import org.openqa.selenium.remote.RemoteWebDriver;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
 
import java.net.MalformedURLException;
import java.net.URL;
 
class MinimizeMultipleBrowserWindow {
 
 
   public String username = "YOUR USERNAME";
   public String accesskey = "YOUR ACCESSKEY";
   public String gridURL = "@hub.lambdatest.com/wd/hub";
 
   @Test
   public void minimizeWindowInChromeBrowser() {
       try {
           RemoteWebDriver driver = null;
           DesiredCapabilities capabilities = new DesiredCapabilities();
           capabilities.setCapability("browserName", "Chrome");
           capabilities.setCapability("version", "latest");
           capabilities.setCapability("platform", "win10"); // If this cap isn't specified, it will just get the any available one
           capabilities.setCapability("build", "MinimizeChromeBrowserWindow");
           capabilities.setCapability("name", "MinimizeChromeBrowserWindowUsingChromeBrowser");
           try {
               driver = new RemoteWebDriver(new URL("https://" + username + ":" + accesskey + gridURL), capabilities);
           } catch (MalformedURLException e) {
               System.out.println("Invalid grid URL");
           } catch (Exception e) {
               System.out.println(e.getMessage());
           }
           System.out.println("Logging into Selenium Playground");
           driver.get("http://labs.lambdatest.com/selenium-playground/");
           driver.manage().window().minimize();
           System.out.println("Minimized the browser window");
           String title=driver.getTitle();
           System.out.println("The title of web page is:"+title);
           driver.close();
           System.out.println("Closing the browser");
       } catch (Exception e) {
 
       }
 
   }
 
   @Test
   public void minimizeWindowInFirefoxBrowser() {
       try {
           RemoteWebDriver driver = null;
           DesiredCapabilities capabilities = new DesiredCapabilities();
           capabilities.setCapability("browserName", "Firefox");
           capabilities.setCapability("version", "94.0");
           capabilities.setCapability("platform", "win10"); // If this cap isn't specified, it will just get the any available one
           capabilities.setCapability("build", "MinimizeFirefoxBrowserWindow");
           capabilities.setCapability("name", "MinimizeFirefoxBrowserWindowUsingFireFoxBrowser");
           try {
               driver = new RemoteWebDriver(new URL("https://" + username + ":" + accesskey + gridURL), capabilities);
           } catch (MalformedURLException e) {
               System.out.println("Invalid grid URL");
           } catch (Exception e) {
               System.out.println(e.getMessage());
           }
           System.out.println("Logging into Selenium Playground");
           driver.get("http://labs.lambdatest.com/selenium-playground/");
           driver.manage().window().minimize();
           System.out.println("Minimized the browser window");
           String title=driver.getTitle();
           System.out.println("The title of web page is:"+title);
           driver.close();
           System.out.println("Closing the browser");
       } catch (Exception e) {
 
       }
 
   }
 
   @Test
   public void minimizeWindowInMicrosoftEdgeBrowser() {
       try {
           RemoteWebDriver driver = null;
           DesiredCapabilities capabilities = new DesiredCapabilities();
           capabilities.setCapability("browserName", "MicrosoftEdge");
           capabilities.setCapability("version", "95.0");
           capabilities.setCapability("platform", "win10"); // If this cap isn't specified, it will just get the any available one
           capabilities.setCapability("build", "MinimizeEdgeBrowserWindow");
           capabilities.setCapability("name", "MinimizeEdgeBrowserWindowUsingEdgeBrowser");
           try {
               driver = new RemoteWebDriver(new URL("https://" + username + ":" + accesskey + gridURL), capabilities);
           } catch (MalformedURLException e) {
               System.out.println("Invalid grid URL");
           } catch (Exception e) {
               System.out.println(e.getMessage());
           }
           System.out.println("Logging into Selenium Playground");
           driver.get("http://labs.lambdatest.com/selenium-playground/");
           driver.manage().window().minimize();
           System.out.println("Minimized the browser window");
           String title=driver.getTitle();
           System.out.println("The title of web page is:"+title);
           driver.close();
           System.out.println("Closing the browser");
       } catch (Exception e) {
 
       }
 
   }
 
 
}

