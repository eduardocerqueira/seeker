//date: 2021-09-24T17:13:12Z
//url: https://api.github.com/gists/1440f06481e4fda2290da2c30d836b91
//owner: https://api.github.com/users/RDayal11

package util;

import org.testng.*;

public class Listener implements ITestListener {

    // This belongs to ITestListener and will execute before the whole Test starts

    @Override
    public void onStart(ITestContext arg0) {
        Reporter.log("About to begin executing Class " + arg0.getName(), true);
    }

    // This belongs to ITestListener and will execute, once the whole Test is finished

    @Override
    public void onFinish(ITestContext arg0) {
        Reporter.log("About to end executing Class " + arg0.getName(), true);
    }
    // This belongs to ITestListener and will execute before each test method

    @Override
    public void onTestStart(ITestResult arg0) {
        Reporter.log("Testcase " + arg0.getName() + " started successfully", true);
    }

    // This belongs to ITestListener and will execute only on the event of successfull test method
    public void onTestSuccess(ITestResult arg0) {
        Reporter.log("Testcase " + arg0.getName() + " passed successfully", true);
    }

    // This belongs to ITestListener and will execute only on the event of fail test

    public void onTestFailure(ITestResult arg0) {
        Reporter.log("Testcase " + arg0.getName() + " failed", true);
    }

    // This belongs to ITestListener and will execute only on the event of skipped test method

    public void onTestSkipped(ITestResult arg0) {
        Reporter.log("Testcase " + arg0.getName() + " got skipped", true);
    }

    @Override
    public void onTestFailedButWithinSuccessPercentage(ITestResult arg0) {
    }
}