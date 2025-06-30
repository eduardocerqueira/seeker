//date: 2025-06-30T16:55:35Z
//url: https://api.github.com/gists/157d33d471342fbf518e457f67e921b8
//owner: https://api.github.com/users/mukel

package com.example.hello;

import com.oracle.truffle.espresso.polyglot.Polyglot;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

public class Main {

    public static void main(String[] args) {
        try (Context context = Context.newBuilder()
                // Just for fun, use the same class path as the host, never do this!!!
                .option("java.Classpath", System.getProperty("java.class.path"))
                .option("java.Polyglot", "true")
                .allowAllAccess(true)
                .build()) {

            // 'java' bindings expose the system/application class loader.
            Value bindings = context.getBindings("java");
            Value main = bindings.getMember("com.example.hello.Main");

            // Call Main.sayHello method in the host first.
            Main.sayHello("Host VM");

            // Then call it within the Espresso context.
            main.invokeMember("sayHello", "Espresso");

            // Show Espresso interop with JS.
            main.invokeMember("espressoInterop");
        }
    }

    public static void sayHello(String toWhom) {
        System.out.println("Hello, " + toWhom + "! from " + System.getProperty("java.vm.name"));
    }

    public static void espressoInterop() {
        // language=js
        String mandelbrot = """                
                const mandelbrot = function(width = 120, height = 30, maxIterations = 25) {
                    let result = "";
                    for (let py = 0; py < height; py++) {
                        let line = "";
                        for (let px = 0; px < width; px++) {
                            // Convert pixel coordinates to complex plane (-2..2)
                            const x0 = (px / width) * 3.5 - 2.5;
                            const y0 = (py / height) * 2 - 1;
                
                            let x = 0;
                            let y = 0;
                            let iteration = 0;
                
                            // Mandelbrot formula: z = z² + c
                            while (x*x + y*y <= 4 && iteration < maxIterations) {
                                const xtemp = x*x - y*y + x0;
                                y = 2*x*y + y0;
                                x = xtemp;
                                iteration++;
                            }
                
                            // Choose character based on iteration count
                            line += iteration === maxIterations ? "■"\s
                                   : iteration > maxIterations * 0.8 ? "#"\s
                                   : iteration > maxIterations * 0.6 ? "*"\s
                                   : iteration > maxIterations * 0.4 ? "+"\s
                                   : iteration > maxIterations * 0.2 ? "."\s
                                   : " ";
                        }
                        result += line + "\\n";
                    }
                    return result;
                }
                mandelbrot()
                """;

        String result = Polyglot.eval("js", mandelbrot, String.class);
        System.out.println(result);
    }
}
