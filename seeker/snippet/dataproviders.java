//date: 2022-04-06T16:57:34Z
//url: https://api.github.com/gists/e8ddca1fa8b6cf3902a0e09c8f80f0da
//owner: https://api.github.com/users/bam018

@DataProvider(name = "sitemap")
public Object[] getSiteMapURLs(){
 String startUrl = "";
 if (currentTestSuite.applicationEnvironment == Environment.TST) {
  startUrl = "https://testerie.erieinsurance.com/sitemap.xml";
 } else if (currentTestSuite.applicationEnvironment == Environment.PRD) {
  startUrl = "https://www.erieinsurance.com/sitemap.xml";
 }
 RestAssured.useRelaxedHTTPSValidation();
 NodeChildrenImpl urlList = RestAssured.given().when()
 .get(startUrl)
 .then().extract().path("urlset.url.loc");
 Object data[] = new Object[urlList.size()];
 for(int i = 0; i < urlList.size(); i ++) {
  data[i] = urlList.get(i).value();
 }
 return data;
}