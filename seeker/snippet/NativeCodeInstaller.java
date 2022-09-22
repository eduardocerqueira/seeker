//date: 2022-09-22T17:14:21Z
//url: https://api.github.com/gists/b726f576334de194adfdb16145d02995
//owner: https://api.github.com/users/xxDark

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.Executable;
import java.lang.reflect.Method;

public class NativeCodeInstaller {

	public static void main(String[] args) throws Throwable {
		MethodHandle code = install(new byte[]{(byte) 0xCC}, new byte[0], 16);
		code.invoke();
	}

	public static MethodHandle install(byte[] assembly, byte[] data, int frameSize) throws Exception {
		Object backend = hotspotBackend();
		Object metaAccess = metaAccess(backend);
		Object codeCache = codeCache(backend);
		Object method = lookupJavaMethod(metaAccess, NativeCodeInstaller.class.getDeclaredMethod("stub"));
		int id = allocateCompileId(method, 0);
		Constructor<?> c = Class.forName("jdk.vm.ci.hotspot.HotSpotCompiledNmethod").getDeclaredConstructors()[0];
		c.setAccessible(true);
		Object compiled = c.newInstance("stub", assembly, assembly.length, emptySite(), emptyAssumption(), resolvedJavaMethodArray(method), emptyComment(), data, 8,
				emptyDataPatch(), false, frameSize, null, method, -1, id, 0L, true);
		Object code = addCode(codeCache, method, compiled);
		return MethodHandles.lookup().findVirtual(code.getClass(), "executeVarargs", MethodType.methodType(Object.class, Object[].class))
				.bindTo(code)
				.asVarargsCollector(Object[].class);
	}

	private static void stub() {
	}

	private static Object hotspotBackend() throws Exception {
		Method m = Class.forName("jdk.vm.ci.runtime.JVMCI").getDeclaredMethod("getRuntime");
		m.setAccessible(true);
		Object rt = m.invoke(null);
		m = m.getReturnType().getDeclaredMethod("getHostJVMCIBackend");
		m.setAccessible(true);
		return m.invoke(rt);
	}

	private static Object metaAccess(Object backend) throws Exception {
		Method m = backend.getClass().getDeclaredMethod("getMetaAccess");
		m.setAccessible(true);
		return m.invoke(backend);
	}

	private static Object codeCache(Object backend) throws Exception {
		Method m = backend.getClass().getDeclaredMethod("getCodeCache");
		m.setAccessible(true);
		return m.invoke(backend);
	}

	private static Object lookupJavaMethod(Object metaAccess, Method method) throws Exception {
		Method m = metaAccess.getClass().getDeclaredMethod("lookupJavaMethod", Executable.class);
		m.setAccessible(true);
		return m.invoke(metaAccess, method);
	}

	private static int allocateCompileId(Object method, int x) throws Exception {
		Method m = method.getClass().getDeclaredMethod("allocateCompileId", Integer.TYPE);
		m.setAccessible(true);
		return (int) m.invoke(method, x);
	}

	private static Object emptySite() throws Exception {
		return Array.newInstance(Class.forName("jdk.vm.ci.code.site.Site"), 0);
	}

	private static Object emptyAssumption() throws Exception {
		return Array.newInstance(Class.forName("jdk.vm.ci.meta.Assumptions$Assumption"), 0);
	}

	private static Object emptyComment() throws Exception {
		return Array.newInstance(Class.forName("jdk.vm.ci.hotspot.HotSpotCompiledCode$Comment"), 0);
	}

	private static Object emptyDataPatch() throws Exception {
		return Array.newInstance(Class.forName("jdk.vm.ci.code.site.DataPatch"), 0);
	}

	private static Object resolvedJavaMethodArray(Object m) throws Exception {
		Object array = Array.newInstance(Class.forName("jdk.vm.ci.meta.ResolvedJavaMethod"), 1);
		((Object[]) array)[0] = m;
		return array;
	}

	private static Object addCode(Object codeCache, Object method, Object compiled) throws Exception {
		Class<?> c = Class.forName("jdk.vm.ci.code.CodeCacheProvider");
		for (Method m : c.getDeclaredMethods()) {
			if ("addCode".equals(m.getName())) {
				m.setAccessible(true);
				return m.invoke(codeCache, method, compiled, null, null);
			}
		}
		throw new IllegalStateException("No addCode method");
	}
}
