//date: 2025-11-17T17:08:11Z
//url: https://api.github.com/gists/6dbd353e0b4de99302190aceeebcb23a
//owner: https://api.github.com/users/guijaci

import java.util.function.Supplier;

// This is a Minimal Reproductible Example for reproducing a javac crash by a NPE using Oracle JDK 21.0.9.
// Just execute `javac TestJavaCompilerNpe.java` with the JDK to receive an error from the compiler:
//```
//An exception has occurred in the compiler (21.0.9). Please file a bug against the Java compiler via the Java bug reporting page (https://bugreport.java.com) after checking the Bug Database (https://bugs.java.com) for duplicates. Include your program, the following diagnostic, and the parameters passed to the Java compiler in your report. Thank you.
//java.lang.NullPointerException: Cannot read field "sym" because "this.lvar[0]" is null
//	at jdk.compiler/com.sun.tools.javac.jvm.Code.emitop0(Code.java:568)
//[...]
//```
// This bug is related to https://bugs.openjdk.org/browse/JDK-8333313, and does not happen using JDK 24.0.2
class TestCaseJavac21NpeCrash {
    public static void main(final String[] args) {
        class Local {
            static Supplier<Local> staticFunction() {
                return Local::new;
            }

            public String[] closure() {
                return args;
            }
        }
    }
}