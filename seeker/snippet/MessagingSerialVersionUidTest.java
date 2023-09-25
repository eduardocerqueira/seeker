//date: 2023-09-25T16:45:07Z
//url: https://api.github.com/gists/d867185288b0f36aa908f45d6225c40c
//owner: https://api.github.com/users/arberg

package dk.messaging;

import com.google.common.truth.Truth;
import com.openpojo.reflection.PojoClass;
import com.openpojo.reflection.impl.PojoClassFactory;

import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;
import org.junit.jupiter.api.function.Executable;

import java.io.ObjectStreamClass;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.DynamicTest.dynamicTest;

// openpojo used in MessagingSerialVersionUidTest for looking up which classes extend another class
// https://mvnrepository.com/artifact/com.openpojo/openpojo
// https://github.com/OpenPojo/openpojo
//    testImplementation("com.openpojo:openpojo:0.9.1")
//    testImplementation("com.google.truth:truth:1.1.3")
//    testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.2")
class MessagingSerialVersionUidTest {

    private static final String parentPackageName = "dk"; // Limit search if you want
    private static final Class<Serializable> parentClass = Serializable.class;

    private static long getSerialVersionUID(Class<?> clazz) {
        return ObjectStreamClass.lookup(clazz).getSerialVersionUID();
    }

    @NotNull
    private static List<PojoClass> getPojoClasses() {
        return PojoClassFactory.enumerateClassesByExtendingType(parentPackageName, parentClass, null)
                .stream()
                .filter(pojoClass -> !pojoClass.isEnum())
                .sorted(Comparator.comparing(p -> p.getClazz().getSimpleName()))
                .collect(Collectors.toList());
    }

    // Generates testCases for all classes with missing serialVersionUID
    @Test
    void generateTestCases() {
        // Get classes and sort them, so we can also order the tests here in the same order, so its easier to figure out which to add

        StringBuilder sb = new StringBuilder();
        for (PojoClass pojoClass : getPojoClasses()) {
            Class<?> clazz = pojoClass.getClazz();
            sb.append(generateTestCaseAndRunTestForClass(clazz, false));
        }
        System.out.println(sb); // All test that failed our test
    }

    @TestFactory
    Collection<DynamicTest> verifySerializedVersionUidExists_TheTestsWillFailWhenWeDetectAMissingUid() throws Exception {
        Collection<DynamicTest> dynamicTests = new ArrayList<>();

        // Get classes and sort them, so we can also order the tests here in the same order, so its easier to figure out which to add
        for (PojoClass pojoClass : getPojoClasses()) {
            Class<?> clazz = pojoClass.getClazz();
            String fullName = clazz.getName();

            String testName = fullName.substring(parentPackageName.length() + 1);
            addDynamicTest(dynamicTests, testName, () -> generateTestCaseAndRunTestForClass(clazz, true));
        }
        return dynamicTests;
    }

    String generateTestCaseAndRunTestForClass(Class<?> clazz, boolean runTest) {
        boolean isSerialVersionUidWhichWeProbablyCreated;
        try {
            clazz.getDeclaredField("serialVersionUID");
            isSerialVersionUidWhichWeProbablyCreated = true;
        } catch (NoSuchFieldException e) {
            isSerialVersionUidWhichWeProbablyCreated = false;
        }
        String simpleName = clazz.getSimpleName();
        long serialVersionUID = getSerialVersionUID(clazz);
        String testCase = "    @Test\n    void TestSerialVersionUid_" + simpleName + "() { \n" +
                "        // private static final long serialVersionUID = " + serialVersionUID + "L;\n" +
                "        Truth.assertThat(getSerialVersionUID(" + simpleName + ".class)).isEqualTo(" + serialVersionUID + "L);\n" +
                "    }";
        if (runTest) {
            System.out.println("Test:");
            System.out.println(testCase);
            Truth.assertThat(isSerialVersionUidWhichWeProbablyCreated).isTrue();
        }
        if (!isSerialVersionUidWhichWeProbablyCreated) {
            return testCase + "\n\n";
        } else {
            return "";
        }
    }


    // These previously had no serialVersionUid, but has been added based on current value

    @Test
    void TestSerialVersionUid_AClassExample() {
        Truth.assertThat(getSerialVersionUID(String.class)).isEqualTo(-6849794470754667710L);
    }

    public static void addDynamicTest(Collection<DynamicTest> dynamicTests, String testName, Executable exec) {
        dynamicTests.add(dynamicTest(testName, exec));
    }
}
