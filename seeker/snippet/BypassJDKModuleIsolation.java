//date: 2023-02-20T16:53:22Z
//url: https://api.github.com/gists/09456b4cbdb70cd934cc0a8caaef2cf8
//owner: https://api.github.com/users/AkaAny

import com.sun.tools.attach.VirtualMachine;
import org.junit.jupiter.api.Test;
import org.openjdk.jol.info.ClassLayout;
import sun.misc.Unsafe;

import java.lang.reflect.*;
import java.util.Arrays;
import java.util.function.Predicate;
import java.util.function.Supplier;

public class AgentTests {
    @Test
    void testAgent() throws Throwable {
        //hotspot(测试环境是zulu-17)对象实例的field在内存里是按照声明顺序排布在object header后面，可以通过Unsafe.objectFieldOffset拿到
        //Field accessibleField= AccessibleObject.class.getDeclaredField("override"); //offset 12
        int overrideFieldOffset = ClassLayout.parseClass(AccessibleObject.class).headerSize() + 0;
        System.out.println("header size:" + overrideFieldOffset);
        Method setAccessible0Method = AccessibleObject.class.getDeclaredMethod("setAccessible0", boolean.class);
        //下面这段注释的代码在JDK17会报错：opens unnamed module
//        directSetAccessibleMethod.setAccessible(true);
//        directSetAccessibleMethod.invoke(field,true);
//        field.set(null,true);
        Constructor<?> ctor = Class.forName("sun.misc.Unsafe").getDeclaredConstructor();
        ctor.setAccessible(true);
        Unsafe unsafe = (Unsafe) ctor.newInstance();
        unsafe.putBoolean((AccessibleObject) (setAccessible0Method), overrideFieldOffset, true);
        //信任边界已经被突破，我们可以任意调用setAccessible0这类protected函数，读取和修改其它模块中私有字段的值
        Field field = Class.forName("sun.tools.attach.HotSpotVirtualMachine").getDeclaredField("ALLOW_ATTACH_SELF");
        setAccessible0Method.invoke(field, true);
        //获取未经过滤的field列表
        Method getDeclaredFields0Method = Class.class.getDeclaredMethod("getDeclaredFields0", boolean.class);
        setAccessible0Method.invoke(getDeclaredFields0Method, true);
        Field[] fieldClassFields = (Field[]) getDeclaredFields0Method.invoke(Field.class, false);
        Field fieldModifiersField = Arrays.stream(fieldClassFields).filter(new Predicate<Field>() {
            @Override
            public boolean test(Field field) {
                return field.getName().equals("modifiers");
            }
        }).findFirst().orElseThrow(new Supplier<Throwable>() {
            @Override
            public Throwable get() {
                return new Throwable("unreachable");
            }
        });
        setAccessible0Method.invoke(fieldModifiersField, true);
        //绕过jdk/internal/reflect/UnsafeQualifiedStaticBooleanFieldAccessorImpl.java:76的限制
        int fieldModifiers = fieldModifiersField.getInt(field);
        fieldModifiers = fieldModifiers & ~Modifier.FINAL; //remove final mark
        fieldModifiersField.set(field, fieldModifiers);
        field.set(null, true);
        //field.setBoolean(null,true);
        long pid = ProcessHandle.current().pid();//ManagementFactory.getRuntimeMXBean().getPid();

        VirtualMachine.attach(Long.toString(pid)).loadAgent("build/libs/agent.jar");
    }
}
