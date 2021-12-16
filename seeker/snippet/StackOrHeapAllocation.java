//date: 2021-12-16T17:16:29Z
//url: https://api.github.com/gists/e616b87452b215db116448216da59636
//owner: https://api.github.com/users/misTrasteos

///usr/bin/env jbang "$0" "$@" ; exit $?

//JAVA_OPTIONS -Xms2m -Xmx2m
//JAVA_OPTIONS -Xlog:gc*
//JAVA_OPTIONS -XX:+UnlockExperimentalVMOptions -XX:+UseEpsilonGC
//JAVA_OPTIONS -XX:+HeapDumpOnOutOfMemoryError

public class StackOrHeapAllocation {

    public static void main(String... args) {

        Integer integerObject = null;
        int integerPrimitive = -1;

        boolean runWithPrimitives = System.getProperty("PRIMITIVE") != null;

        while(args.length > -1) // something that returns always true
            if(runWithPrimitives)
                integerPrimitive = getIntegerPrimitive();
            else
                integerObject = getIntegerObject();

        System.out.println( integerPrimitive );
        System.out.println( integerObject );
    }
    
    private static Integer getIntegerObject(){
        return new Integer(0);
    }

    private static int getIntegerPrimitive(){
        return 0;
    }

}
