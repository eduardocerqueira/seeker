//date: 2021-12-10T17:06:00Z
//url: https://api.github.com/gists/08c482ba264f20098f9cf3bbe9af65f0
//owner: https://api.github.com/users/dez011

import org.testng.Assert;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class ParamTestWithDataProvider1 {
   private PrimeNumberChecker primeNumberChecker;

   @BeforeMethod
   public void initialize() {
      primeNumberChecker = new PrimeNumberChecker();
   }

   @DataProvider(name = "test1")
   public static Object[][] primeNumbers() {
      return new Object[][] {{2, true}, {6, false}, {19, true}, {22, false}, {23, true}};
   }

   // This test will run 4 times since we have 5 parameters defined
   @Test(dataProvider = "test1")
   public void testPrimeNumberChecker(Integer inputNumber, Boolean expectedResult) {
      System.out.println(inputNumber + " " + expectedResult);
      Assert.assertEquals(expectedResult, primeNumberChecker.validate(inputNumber));
   }
}


//testng.xml /work/testng/src
<?xml version = "1.0" encoding = "UTF-8"?>
<!DOCTYPE suite SYSTEM "http://testng.org/testng-1.0.dtd" >

<suite name = "Suite1">
   <test name = "test1">
      <classes>
         <class name = "ParamTestWithDataProvider1" />
      </classes>
   </test>
</suite>


