//date: 2024-11-18T16:53:29Z
//url: https://api.github.com/gists/c64612f8a3d4daa1b9d5a98758acee59
//owner: https://api.github.com/users/mmerrell

  public MutableCapabilities getChromeDesktopCapabilities() {
    ChromeOptions browserOptions = new ChromeOptions();
    browserOptions.setPlatformName(this.os);
    browserOptions.setBrowserVersion("latest");
    Map<String, Object> caps = new HashMap<>();
    caps.put("projectName", "Loom");
    caps.put("browserVersion", "latest");
    caps.put("name", getTestInfo().getTestClass().get().getSimpleName() + "." + getTestInfo().getTestMethod().get().getName());
    browserOptions.setCapability("sauce:options", caps);
    log.info("Sauce Labs Capabilities: " + browserOptions);
    return browserOptions;
  }

  public void pass(WebDriver driver) {
    ((JavascriptExecutor) driver).executeScript("sauce:job-result=passed");
  }


  public void fail(WebDriver driver, Throwable cause) {
    ((JavascriptExecutor) driver).executeScript("sauce:job-result=failed");
    ((JavascriptExecutor) driver).executeScript("sauce:context=Failure Reason: " + cause.getMessage());

    for (Object trace : Arrays.stream(cause.getStackTrace()).toArray()) {
      if (trace.toString().contains("sun")) {
        break;
      }
      ((JavascriptExecutor) driver).executeScript("sauce:context=" + trace);
    }
  }