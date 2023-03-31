//date: 2023-03-31T17:06:45Z
//url: https://api.github.com/gists/6a54f08f78735ddfdec2622dc74c17fd
//owner: https://api.github.com/users/wanted0

import android.app.ActivityThread;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.os.Looper;
import java.util.Iterator;

public class Main {
   public static void main(String[] var0) {
      Looper.prepareMainLooper();
      PackageManager var1 = ActivityThread.systemMain().getSystemContext().getPackageManager();
      Iterator var2 = var1.getInstalledApplications(8192).iterator();

      while(var2.hasNext()) {
         ApplicationInfo var3 = (ApplicationInfo)var2.next();
         System.out.println(var3.uid + " " + var3.packageName + " " + var1.getApplicationLabel(var3));
      }

   }
}
