//date: 2022-07-15T17:22:20Z
//url: https://api.github.com/gists/2e877204955a662aa6bdc39a94e1a439
//owner: https://api.github.com/users/SarahElson

package com.lambdatest.appium.sample.pages;
 
import java.util.Map;
 
import com.google.common.collect.ImmutableMap;
import com.lambdatest.appium.sample.enums.Platform;
import io.appium.java_client.MobileBy;
import org.openqa.selenium.By;
 
public class HomePage {
   // 1.
   public Map<Platform, By> message () {
       return ImmutableMap.of (Platform.IOS, MobileBy.AccessibilityId ("Textbox"), Platform.ANDROID,
           By.id ("Textbox"));
   }
 
   // 2.
   public Map<Platform, By> notificationButton () {
       return ImmutableMap.of (Platform.IOS, MobileBy.AccessibilityId ("notification"), Platform.ANDROID,
           By.id ("notification"));
   }
 
   // 3.
   public Map<Platform, By> proverbialNotification () {
       return ImmutableMap.of (Platform.IOS, MobileBy.iOSNsPredicateString ("label BEGINSWITH \"PROVERBIAL\""),
           Platform.ANDROID, By.id ("android:id/title"));
   }
 
   // 4.
   public Map<Platform, By> textButton () {
       return ImmutableMap.of (Platform.IOS,
           MobileBy.iOSNsPredicateString ("label == \"Text\" AND type == \"XCUIElementTypeButton\""), Platform.ANDROID,
           By.id ("Text"));
   }
}
