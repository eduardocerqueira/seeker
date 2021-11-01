//date: 2021-11-01T17:19:12Z
//url: https://api.github.com/gists/ff153ebc5881a2fc245e938681142de1
//owner: https://api.github.com/users/7h3kk1d

package com.thekkid.ninetyninelambdas.problems;

import com.jnape.palatable.lambda.adt.Unit;
import com.jnape.palatable.lambda.functions.builtin.fn2.ToCollection;
import com.jnape.palatable.lambda.functor.builtin.State;
import com.jnape.palatable.lambda.io.IO;
import com.jnape.palatable.lambda.monad.MonadRec;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

import static com.jnape.palatable.lambda.functions.builtin.fn2.Cons.cons;
import static com.jnape.palatable.lambda.functor.builtin.State.state;
import static com.jnape.palatable.lambda.io.IO.io;

public class Echo<M extends MonadRec<?, M>> {
    Input<M> input;
    Output<M> output;

    public Echo(Input<M> input, Output<M> output) {
        this.input = input;
        this.output = output;
    }

    private interface Input<M extends MonadRec<?, M>> {
        MonadRec<String, M> read();
    }

    private interface Output<M extends MonadRec<?, M>> {
        MonadRec<Unit, M> write(String string);
    }

    public <MU extends MonadRec<Unit, M>> MU echo() {
        return input.read().flatMap(s -> output.write(s)).coerce();
    }

    public static class IoInput implements Input<IO<?>> {

        @Override
        public IO<String> read() {
            return io(() -> new Scanner(System.in).nextLine());
        }
    }

    public static class IoOutput implements Output<IO<?>> {
        @Override
        public MonadRec<Unit, IO<?>> write(String string) {
            return io(() -> System.out.println(string));
        }
    }

    public static class StateInput implements Input<State<List<String>, ?>> {
        String input;

        public StateInput(String input) {
            this.input = input;
        }

        @Override
        public State<List<String>, String> read() {
            return state(input);
        }
    }

    public static class StateOutput implements Output<State<List<String>, ?>> {
        @Override
        public State<List<String>, Unit> write(String string) {
            return State.modify(output -> ToCollection.toCollection(ArrayList::new, cons(string, output)));
        }
    }


    public static void main(String[] args) {
        // Echo with IO
        Echo<IO<?>> ioEcho = new Echo<>(new IoInput(), new IoOutput());
        IO<Unit> io = ioEcho.echo();
        io.unsafePerformIO();

        // Echo with state and no IO
        Echo<State<List<String>, ?>> stateEcho = new Echo<>(new StateInput("hardcoded input"), new StateOutput());
        State<List<String>, Unit> state = stateEcho.echo();
        List<String> output = state.exec(Collections.emptyList());
        System.out.println(output);
    }
}
