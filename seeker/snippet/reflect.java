//date: 2021-10-01T17:07:41Z
//url: https://api.github.com/gists/d0fb4ecbd3b9e8ed969987ee753bbe85
//owner: https://api.github.com/users/18626428291

package clazz.reflect;

import clazz.loader.ClassLoaderTest;

import java.lang.annotation.Annotation;
import java.lang.annotation.Repeatable;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.Arrays;

/**
 * Created by xujiajun on 2021/10/2
 */
@Repeatable(Annos.class)
@interface Anno {
}

@Retention(value = RetentionPolicy.RUNTIME)
@interface Annos {
	Anno[] value();
}

@SuppressWarnings(value = "unchecked")
@Anno
@Anno
@Deprecated
public class ClassTest {
	private ClassTest() {
	}

	public ClassTest(String name) {
		System.out.println("这是一个有参构造器");
	}

	public void info() {
		System.out.println("无参数info方法");
	}

	public void info(String name) {
		System.out.println("有参数info方法,参数是：" + name);
	}

	//内部类
	class inner {

	}

	public static void main(String[] args) throws ClassNotFoundException, NoSuchMethodException {
		Class<ClassTest> clazz = ClassTest.class;
		//获取构造器
		Constructor[] c = clazz.getDeclaredConstructors();
		System.out.println("全部构造器：");
		for (Constructor constructor : c) {
			System.out.println(constructor);
		}
		//获取公开的构造器
		c = clazz.getConstructors();
		System.out.println("公开构造器");
		for (Constructor constructor : c) {
			System.out.println(constructor);
		}
		//公开方法
		Method[] m = clazz.getMethods();
		System.out.println("公开方法");
		for (Method method : m) {
			System.out.println(method);
		}
		//获取指定方法
		System.out.println("参数string的info方法：" + clazz.getMethod("info", String.class));
		//获取全部注解
		Annotation[] a = clazz.getAnnotations();
		System.out.println("全部注解：");
		for (Annotation annotation : a) {
			System.out.println(annotation);
		}
		//指定注解
		System.out.println("anno注解是：" + Arrays.toString(clazz.getAnnotationsByType(Anno.class)));
		//内部类
		Class<?>[] innerclass = clazz.getDeclaredClasses();
		System.out.println("内部类");
		for (Class<?> aClass : innerclass) {
			System.out.println(aClass);
		}
		//使用Class.forname加载innerClass
		Class<?> inner = Class.forName("clazz.reflect.ClassTest$inner");
		//外部类
		System.out.println("inner的外部类是：" + inner.getDeclaringClass());
		//b包
		System.out.println("class所在包：" + inner.getPackageName());
		//父类
		System.out.println("父类：" + inner.getSuperclass());
	}
}


