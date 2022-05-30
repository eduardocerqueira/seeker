//date: 2022-05-30T17:18:50Z
//url: https://api.github.com/gists/e7b46947be7f7de466a1123374393884
//owner: https://api.github.com/users/cuteofdragon

public class AppInfo {

    /**
     * Get application version string
     * 
     * @param context
     *            Interface to global information about an application
     *            environment
     * @return String of the form "VV-CC", where VV is the version name and CC
     *         is the version code (e.g., "1.0.3-25")
     */
    public static String getAppVersionNameAndVersionCode(Context context) {
        try {
            PackageManager pm = context.getPackageManager();
            String packageName = context.getPackageName();
            PackageInfo info = pm.getPackageInfo(packageName, 0);

            String version = info.versionName + "-" + info.versionCode;
            return version;
        }
        catch (Exception ex) {
            return null;
        }
    }
}