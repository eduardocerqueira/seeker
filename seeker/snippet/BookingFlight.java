//date: 2024-01-16T17:01:52Z
//url: https://api.github.com/gists/a33abe653800d37b33fccf7d13974471
//owner: https://api.github.com/users/fahadPathan7

package org.example;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.firefox.FirefoxDriver;

public class BookingFlight {

    WebDriver webDriver;

    public void loadWeb() throws InterruptedException {
        System.setProperty("webdriver.gecko.driver", "C:\\Users\\User\\Downloads\\geckodriver-v0.34.0-win64\\geckodriver.exe");
        webDriver = new FirefoxDriver();
        webDriver.get("https://book.spicejet.com/search.aspx");
        //webDriver.navigate().to("https://book.spicejet.com/search.aspx");
        Thread.sleep(2000);
    }

    public void selectDepartureCity() throws InterruptedException {
        webDriver.findElement(By.id("ControlGroupSearchView_AvailabilitySearchInputSearchVieworiginStation1_CTXT")).click();
        Thread.sleep(2000);
        webDriver.findElement(By.linkText("Chennai (MAA)")).click();
        Thread.sleep(2000);
    }

    public void selectArrivalCity() throws InterruptedException {
        webDriver.findElement(By.id("ControlGroupSearchView_AvailabilitySearchInputSearchViewdestinationStation1_CTXT")).click();
        Thread.sleep(2000);
        webDriver.findElement(By.linkText("Delhi (DEL)")).click();
        Thread.sleep(2000);
    }

    public void departureDate() throws InterruptedException {
        webDriver.findElement(By.id("custom_date_picker_id_1")).click();
        Thread.sleep(2000);
        webDriver.findElement(By.linkText("20")).click();
        Thread.sleep(2000);
    }

    public void currencySelect() throws InterruptedException {
        webDriver.findElement(By.id("ControlGroupSearchView_AvailabilitySearchInputSearchView_DropDownListCurrency")).click();
        Thread.sleep(2000);
        webDriver.findElement(By.cssSelector("option[value='BDT']")).click();
        Thread.sleep(2000);
    }

    public void searchFlight() throws InterruptedException {
        webDriver.findElement(By.id("ControlGroupSearchView_AvailabilitySearchInputSearchView_ButtonSubmit")).click();
        Thread.sleep(2000);
    }

    public void exit() {
        webDriver.quit();
    }

    public static void main(String[] args) throws InterruptedException {
        BookingFlight bookingFlight = new BookingFlight();
        bookingFlight.loadWeb();
        bookingFlight.selectDepartureCity();
        bookingFlight.selectArrivalCity();
        bookingFlight.departureDate();
        bookingFlight.currencySelect();
        bookingFlight.searchFlight();
        //bookingFlight.exit();
    }


}
