//date: 2023-05-11T16:51:02Z
//url: https://api.github.com/gists/f4592f08e5b68774ea75a0c96113d9eb
//owner: https://api.github.com/users/avozme


import com.googlecode.lanterna.TerminalPosition;
import com.googlecode.lanterna.TerminalSize;
import com.googlecode.lanterna.TextColor;
import com.googlecode.lanterna.screen.Screen;
import com.googlecode.lanterna.screen.TerminalScreen;
import com.googlecode.lanterna.terminal.DefaultTerminalFactory;
import com.googlecode.lanterna.terminal.Terminal;

public class HelloWorldLanterna {

    public static void main(String[] args) {
        try {
            Terminal terminal = new DefaultTerminalFactory().createTerminal();
            Screen screen = new TerminalScreen(terminal);

            screen.startScreen();
            screen.setCursorPosition(null);

            TerminalSize terminalSize = screen.getTerminalSize();
            int columns = terminalSize.getColumns();
            int rows = terminalSize.getRows();

            // Calcular las posiciones para mostrar el texto en diferentes lugares
            TerminalPosition position1 = new TerminalPosition(columns / 2 - 5, rows / 2 - 1);
            TerminalPosition position2 = new TerminalPosition(columns - 12, rows - 3);
            TerminalPosition position3 = new TerminalPosition(1, 1);

            screen.putString(position1, "Hola mundo", TextColor.ANSI.WHITE, TextColor.ANSI.BLACK);
            screen.putString(position2, "Hola mundo", TextColor.ANSI.RED, TextColor.ANSI.BLACK);
            screen.putString(position3, "Hola mundo", TextColor.ANSI.GREEN, TextColor.ANSI.BLACK);

            screen.refresh();

            Thread.sleep(2000);

            screen.stopScreen();
            terminal.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}