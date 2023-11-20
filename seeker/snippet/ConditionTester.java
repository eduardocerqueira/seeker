//date: 2023-11-20T17:08:36Z
//url: https://api.github.com/gists/dafa330947d3ed7b48e7fe0a2cb1f995
//owner: https://api.github.com/users/nadvolod

package com.saucelabs.saucebindings.junit4.examples;

public class ConditionTester {
    private boolean actionDone;

    public boolean option1(boolean conditionA, boolean conditionB) {
        if (!conditionA) {
            actionDone = true; // Represents "Another action"
            return actionDone;
        }

        if (conditionB) {
            actionDone = true; // Represents "Do something"
        } else {
            actionDone = true; // Represents "Alternative action"
        }
        return actionDone;
    }

    public boolean option2(boolean conditionA, boolean conditionB) {
        if (conditionA) {
            if (conditionB) {
                actionDone = true; // Represents "Do something"
            } else {
                actionDone = true; // Represents "Alternative action"
            }
        } else {
            actionDone = true; // Represents "Another action"
        }
        return actionDone;
    }
}
