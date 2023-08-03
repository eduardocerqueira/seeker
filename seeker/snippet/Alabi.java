//date: 2023-08-03T16:49:32Z
//url: https://api.github.com/gists/ba28e0ccfa0cf726a2bdeb5861debaa3
//owner: https://api.github.com/users/lurldgbodex

public class Alabi {
    private String[] callLog;
    private static boolean canModifyCallLog;

    public Alabi(){
        this.callLog = new String[] {"contact List 1", "contact List 2"};
    }

    public String[] getCallLog(){
        StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
        String callerClass = null;
        if (stackTrace.length >= 3) {
            callerClass = stackTrace[2].getClassName();
            boolean canAccessCallLog = callerClass.equals("Wife") || callerClass.equals("Daughter");
            if (canAccessCallLog) {
                return this.callLog;
            }
        }
        return new String[]{callerClass + "cannot access callLog"};
    }

    public void setCallLog() {
        StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();

        String callerClass = null;
        if (stackTrace.length >= 3) {
            callerClass = stackTrace[2].getClassName();
            canModifyCallLog = callerClass.equals("Wife");
        }

        if (canModifyCallLog) {
            System.out.println(callerClass + " changing callLog");
            this.callLog = new String[] {"call log modified"};
        } else {
            System.out.println("Access denied. " + callerClass + " cannot modify callLog");
        }

    }
}