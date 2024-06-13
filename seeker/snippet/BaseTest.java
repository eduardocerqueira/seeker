//date: 2024-06-13T16:50:42Z
//url: https://api.github.com/gists/b04db1f40db2c4505a5473d042353cda
//owner: https://api.github.com/users/SarahElson

package LocalGrid;


import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.testng.annotations.*;


public class BaseTest {


   public static WebDriver driver;
  
   String username = "admin";
   String password = "**********"


   @BeforeClass
   public void setDriver() {
       driver = new ChromeDriver();
   }


   @AfterClass
   public void tearDown() {
       driver.quit();
   }
}
