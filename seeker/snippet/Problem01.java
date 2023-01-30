//date: 2023-01-30T16:40:40Z
//url: https://api.github.com/gists/d57c9bc6ac6ae242f5292634511d988d
//owner: https://api.github.com/users/toksaitov

class Turtle {
    // Turtle's State
    int x, y;
    int dx, dy;
    boolean penDown;

    // Turtle's Services/Behaviour
    Turtle() {
        x = y = 0;
        dx = 1;
        dy = 0;
        penDown = false;
    }

    void putPenUp() {
        penDown = false;
    }

    void putPenDown() {
        penDown = true;
    }

    void turnRight() {
        int temp = dx;
        dx = -dy;
        dy = temp;
    }

    void turnLeft() {
        int temp = dx;
        dx = dy;
        dy = -temp;
    }

    void move(int steps) {
        for (int i = 0; i < steps; ++i) {
            int nextX = x + dx;
            int nextY = y + dy;
            // TODO: fix the code below
            // if (!areCoordsInsideField(nextX, nextY)) {
            //    break;
            // }

            if (penDown) {
                // TODO: fix the code below
                // markField(x, y);
            }

            x = nextX;
            y = nextY;
        }
    }
}

import java.util.Scanner;

public class Problem01 {
    // Field's State
    static final char EMPTY_CELL  = '.';
    static final char MARKED_CELL = '*';
    static final char TURTLE_CELL = 'T';
    static final int FIELD_WIDTH  = 20;
    static final int FIELD_HEIGHT = 20;
    static char[][] field;

    // Field's Services/Behaviour
    static void constructField() {
        field = new char[FIELD_HEIGHT][FIELD_WIDTH];
        for (int y = 0; y < FIELD_HEIGHT; ++y) {
            for (int x = 0; x < FIELD_WIDTH; ++x) {
                field[y][x] = EMPTY_CELL;
            }
        }
    }

    static void displayField(Turtle[] turtles) {
        for (int y = 0; y < FIELD_HEIGHT; ++y) {
            columnLoop:
            for (int x = 0; x < FIELD_WIDTH; ++x) {
                for (Turtle turtle : turtles) {
                    if (x == turtle.x && y == turtle.y) {
                        System.out.print(TURTLE_CELL);
                        continue columnLoop;
                    }
                }
                System.out.print(field[y][x]);
            }
            System.out.println();
        }
    }

    static boolean areCoordsInsideField(int x, int y) {
        return x >= 0 && x < FIELD_WIDTH &&
               y >= 0 && y < FIELD_HEIGHT;
    }

    static void markField(int x, int y) {
        if (areCoordsInsideField(x, y)) {
            field[y][x] = MARKED_CELL;
        }
    }

    public static void main(String[] args) {
        constructField();
        Turtle turtle1 = new Turtle();
        Turtle turtle2 = new Turtle();
        Turtle[] turtles = new Turtle[] { turtle1, turtle2 };
        Turtle turtle = turtle1;

        Scanner scanner = new Scanner(System.in);
        while (scanner.hasNext()) {
            String input = scanner.nextLine(); // "turn-left", "move 12"
            String[] parts = input.split(" "); // { "turn-left" }, { "move", "12" }
            if (!(parts.length > 0 && parts[0].trim().length() > 0)) {
                // TODO: print an error message
                continue;
            }
            String command = parts[0].trim();
            switch (command) {
                case "select":
                    if (!(parts.length == 2 && parts[1].trim().length() > 0)) {
                        // TODO: print an error message
                        continue;
                    }
                    int turtleNum = Integer.parseInt(parts[1]); // TODO: handle errors
                    if (turtleNum <= 0 || turtleNum > turtles.length) {
                        // TODO: print an error message
                        continue;
                    }
                    turtle = turtles[turtleNum - 1];
                    break;
                case "pen-up":
                    turtle.putPenUp();
                    break;
                case "pen-down":
                    turtle.putPenDown();
                    break;
                case "turn-right":
                    turtle.turnRight();
                    break;
                case "turn-left":
                    turtle.turnLeft();
                    break;
                case "move":
                    if (!(parts.length == 2 && parts[1].trim().length() > 0)) {
                        // TODO: print an error message
                        continue;
                    }
                    int steps = Integer.parseInt(parts[1]); // TODO: handle errors
                    if (steps < 0) {
                        // TODO: print an error message
                        continue;
                    }
                    turtle.move(steps);
                    break;
                case "display":
                    displayField(turtles);
                    break;
                case "exit":
                    System.exit(0);
                    break;
                default:
                    // TODO: print an error message
                    break;
            }
        }
    }
}
