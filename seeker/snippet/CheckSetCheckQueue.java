//date: 2024-06-26T17:09:51Z
//url: https://api.github.com/gists/38ef90dd45cecf6a8db35174fa4d7aff
//owner: https://api.github.com/users/m1l4n54v1c

package io.event.thinking;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

public class CheckSetCheck {

    private final Queue<Runnable> queue = new ConcurrentLinkedQueue<>();

    public void execute(Runnable item) {
        queue.add(item);
    }
}