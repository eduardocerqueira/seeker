//date: 2023-05-11T16:57:18Z
//url: https://api.github.com/gists/b9324fd855a5c4a69d88822761de17ee
//owner: https://api.github.com/users/avozme

import com.googlecode.lanterna.TerminalSize;
import com.googlecode.lanterna.TextColor;
import com.googlecode.lanterna.input.KeyStroke;
import com.googlecode.lanterna.input.KeyType;
import com.googlecode.lanterna.screen.Screen;
import com.googlecode.lanterna.screen.TerminalScreen;
import com.googlecode.lanterna.terminal.DefaultTerminalFactory;
import com.googlecode.lanterna.terminal.Terminal;

public class KeyInputLanterna {

    public static void main(String[] args) {
        try {
            Terminal terminal = new DefaultTerminalFactory().createTerminal();
            Screen screen = new TerminalScreen(terminal);
            screen.startScreen();
            screen.setCursorPosition(null);
            TerminalSize terminalSize = screen.getTerminalSize();
            int columns = terminalSize.getColumns();
            int rows = terminalSize.getRows();

            boolean running = true;
            while (running) {
                KeyStroke keyStroke = screen.pollInput();
                if (keyStroke != null) {
                    if (keyStroke.getKeyType() == KeyType.Character) {
                        char keyChar = keyStroke.getCharacter();
                        screen.clear();
                        screen.putString(columns / 2 - 10, rows / 2, "Has pulsado la tecla: " + keyChar,
                                TextColor.ANSI.WHITE, TextColor.ANSI.BLACK);
                        screen.refresh();
                    } else if (keyStroke.getKeyType() == KeyType.Escape) {
                        running = false;
                    }
                }
            }

            screen.stopScreen();
            terminal.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
