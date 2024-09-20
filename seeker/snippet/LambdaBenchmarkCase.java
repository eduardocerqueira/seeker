//date: 2024-09-20T16:55:22Z
//url: https://api.github.com/gists/088b55f552353f71e71a5e34f6dfdef3
//owner: https://api.github.com/users/dreamlike-ocean

package io.github.dreamlike.stableValue.Benchmark;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.lang.classfile.AccessFlags;
import java.lang.classfile.ClassFile;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.DynamicCallSiteDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.*;
import java.lang.reflect.AccessFlag;
import java.lang.reflect.Method;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

import static java.lang.constant.ConstantDescs.*;

@State(Scope.Benchmark)
@Warmup(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@BenchmarkMode(Mode.Throughput)
@Threads(value = 5)
@Measurement(iterations = 2, time = 200, timeUnit = TimeUnit.MILLISECONDS)
public class LambdaBenchmarkCase {
//Benchmark                                                        Mode  Cnt            Score            Error  Units
//stableValue.Benchmark.LambdaBenchmarkCase.testCallSite          thrpt   10  24255154361.137 ± 1457667464.139  ops/s
//stableValue.Benchmark.LambdaBenchmarkCase.testConst             thrpt   10  24519036075.193 ±  928404771.651  ops/s
//stableValue.Benchmark.LambdaBenchmarkCase.testDirect            thrpt   10  24727533894.822 ±  592544888.327  ops/s
//stableValue.Benchmark.LambdaBenchmarkCase.testLazy              thrpt   10   2835881869.811 ±  213543161.668  ops/s
//stableValue.Benchmark.LambdaBenchmarkCase.testLmf               thrpt   10  13112492685.481 ± 1007502930.123  ops/s
//stableValue.Benchmark.LambdaBenchmarkCase.testLmfInvokeDynamic  thrpt   10  13102632461.980 ±  564969148.037  ops/s
//stableValue.Benchmark.LambdaBenchmarkCase.testReflect           thrpt   10  10570293119.587 ±  214468018.588  ops/s
//stableValue.Benchmark.LambdaBenchmarkCase.testReflectLazy       thrpt   10   1473938487.388 ±   22329375.514  ops/s
    private final static MethodHandle ADD_MH;
    private final static Method METHOD;
    private static final MutableCallSite callSite = new MutableCallSite(
            MethodType.methodType(int.class, int.class, int.class)
    );
    private static final MethodHandle callSiteInvoke = callSite.dynamicInvoker();

    static {
        try {
            ADD_MH = MethodHandles.lookup().findStatic(Math.class, "addExact", MethodType.methodType(int.class, int.class, int.class));
            METHOD = Math.class.getMethod("addExact", int.class, int.class);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    private final Method LazyMethod;

    private final MethodHandle ADD_LAZY_MH;

    private final Add add;

    private final Add addInvokeDynamic;

    public LambdaBenchmarkCase() {
        callSite.setTarget(ADD_MH);
        ADD_LAZY_MH = ADD_MH;
        LazyMethod = METHOD;

        try {
            add = (Add) LambdaMetafactory.metafactory(
                    MethodHandles.lookup(),
                    "apply",
                    MethodType.methodType(Add.class),
                    MethodType.methodType(int.class, int.class, int.class),
                    ADD_MH,
                    ADD_MH.type()
            ).getTarget().invokeExact();

            addInvokeDynamic = generate(() -> ADD_MH);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
        VarHandle.storeStoreFence();
    }

    public static Add generate(Supplier<MethodHandle> supplier) throws IllegalAccessException, InstantiationException {
        byte[] classByteCode = ClassFile.of()
                .build(ClassDesc.of(LambdaBenchmarkCase.class.getName() + "AddImpl"), cb -> {
                    cb.withInterfaceSymbols(Add.class.describeConstable().get());
                    cb.withMethodBody(ConstantDescs.INIT_NAME, ConstantDescs.MTD_void, AccessFlags.ofMethod(AccessFlag.PUBLIC).flagsMask(), it -> {
                        it.aload(0);
                        it.invokespecial(CD_Object, INIT_NAME, MTD_void);
                        it.return_();
                    });
                    cb.withMethodBody("apply",
                            MethodTypeDesc.of(CD_int, CD_int, CD_int),
                            AccessFlags.ofMethod(AccessFlag.PUBLIC, AccessFlag.SYNTHETIC).flagsMask(),
                            it -> {
                                it.iload(1);
                                it.iload(2);
                                it.invokeDynamicInstruction(
                                        DynamicCallSiteDesc.of(
                                                ConstantDescs.ofCallsiteBootstrap(LambdaBenchmarkCase.class.describeConstable().get(), "indyLambdaFactory", ConstantDescs.CD_CallSite),
                                                "apply",
                                                MethodTypeDesc.of(CD_int, CD_int, CD_int)
                                        )
                                );
                                it.returnInstruction(TypeKind.IntType);
                            });
                });

        MethodHandles.Lookup lookup = MethodHandles.lookup()
                .defineHiddenClassWithClassData(classByteCode, supplier, true);

        return (Add) lookup.lookupClass().newInstance();
    }

    public static CallSite indyLambdaFactory(MethodHandles.Lookup lookup, String name, MethodType type) throws NoSuchFieldException, IllegalAccessException {
        MethodHandle methodHandle = ((Supplier<MethodHandle>) MethodHandles.classData(lookup, ConstantDescs.DEFAULT_NAME, Supplier.class)).get();
        return new ConstantCallSite(methodHandle);
    }

    @Benchmark
    public void testDirect(Blackhole bh) {
        bh.consume(Math.addExact(1, 2));
    }

    @Benchmark
    public void testLazy(Blackhole bh) throws Throwable {
        bh.consume((int) ADD_LAZY_MH.invokeExact(1, 2));
    }

    @Benchmark
    public void testCallSite(Blackhole bh) throws Throwable {
        bh.consume((int) callSiteInvoke.invokeExact(1, 2));
    }

    @Benchmark
    public void testConst(Blackhole bh) throws Throwable {
        bh.consume((int) ADD_MH.invokeExact(1, 2));
    }

    @Benchmark
    public void testReflect(Blackhole bh) throws Throwable {
        bh.consume((int) METHOD.invoke(null, 1, 2));
    }

    @Benchmark
    public void testReflectLazy(Blackhole bh) throws Throwable {
        bh.consume((int) LazyMethod.invoke(null, 1, 2));
    }

    @Benchmark
    public void testLmf(Blackhole bh) throws Throwable {
        bh.consume(add.apply(1, 2));
    }

    @Benchmark
    public void testLmfInvokeDynamic(Blackhole bh) throws Throwable {
        bh.consume(addInvokeDynamic.apply(1, 2));
    }

    interface Add {
        int apply(int a, int b);
    }
}
