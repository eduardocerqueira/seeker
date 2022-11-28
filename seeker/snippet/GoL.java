//date: 2022-11-28T17:07:39Z
//url: https://api.github.com/gists/4c19d39dfb14caaf6d40ed9ee7b6fb75
//owner: https://api.github.com/users/denkspuren

class GoL {
    int width, height;
    int[] world;

    GoL(int width, int height) {
        assert width >= 1 && height >= 1;
        this.width = width + 2;   // ergänze "unsichtbaren" Rand links und rechts
        this.height = height + 2; // ergänze "unsichtbaren" Rand oben und unten
        this.world = new int[this.width * this.height];
    }

    GoL set(int row, int col) {
        assert row >= 1 && row < height - 1 : "1 <= row < " + (height - 1);
        assert col >= 1 && col < width - 1 : "1 <= col < " + (width - 1);
        world[row * width + col] = 1;
        return this;
    }

    int rule(int center) {
        int[] neighbours = new int[]{center - 1, center + 1, center + width, center - width, // left, right, above, below
                                     center - width - 1, center - width + 1,                 // top left/right
                                     center + width - 1, center + width + 1};                // bottom left/right
        int LIVE = 1, DEAD = 0, count = 0;
        for (int pos : neighbours) count += world[pos];
        if (world[center] == DEAD) return count == 3 ? LIVE : DEAD;
        if (count == 2 || count == 3) return LIVE;
        return DEAD; // underpopulation: count < 2, overpopulation: count > 3
    }

    void timestep() {
        int[] newWorld = new int[world.length];
        for (int row = 1; row < height - 1; row++) {
            for (int col = 1; col < width - 1; col++) {
                int pos = row * width + col;
                newWorld[pos] = rule(pos);
            }
        }
        world = newWorld;
    }

    @Override public String toString() {
        String s = "";
        char[] symbols = new char[]{'.', '*'};
        for (int row = 1; row < height - 1; row++) {
            for (int col = 1; col < width - 1; col++) {
                int pos = row * width + col;
                s += symbols[world[pos]];
            }
            s += row != height ? "\n" : "";
        }
        return s;
    }

    void run(int steps) {
        while (steps-- >= 1) {
            timestep();
            System.out.println(this);
        }
    }

    GoL insert(int row, int col, GoL source) {
        for (int y = 1; y < source.height - 1; y++) {
            for (int x = 1; x < source.width - 1; x++) {
                int sourcePos = y * source.width + x;
                int targetRef = (row - 1) * width + (col - 1);
                int targetPos = targetRef + y * width + x;
                world[targetPos] = source.world[sourcePos];
            }
        }
        return this;
    }
}

GoL block = new GoL(2,2).set(1,1).set(1,2).set(2,1).set(2,2);
GoL boat = new GoL(3,3).set(1,1).set(1,2).set(2,1).set(2,3).set(3,2);
GoL blinker = new GoL(3,1).set(1,1).set(1,2).set(1,3);
GoL glider = new GoL(3,3).set(1,3).set(2,3).set(3,3).set(2,1).set(3,2);

/*

Achtung: Die JShell verkürzt Ausgaben auf der Konsole, d.h. das Spielfeld von
GoL wird durch `toString` möglicherweise unvollständig angezeigt.
Spielfeldausgaben sollten ein `System.out.println` verwenden.

 ~~~
jshell> GoL g = new GoL(10,10)
g ==> ..........
..........
..........
..........
..... ... ...
..........
..........


jshell> System.out.println(g)
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
 ~~~
 */

GoL smallWorld = new GoL(20,15).insert(2,1,glider);
// smallWorld.run(20)
